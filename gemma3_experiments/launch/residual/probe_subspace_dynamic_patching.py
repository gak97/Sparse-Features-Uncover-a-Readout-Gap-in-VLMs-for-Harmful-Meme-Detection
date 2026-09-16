

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import torch
import torch.nn as nn
from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.residual_sae.gemma import (
    GemmaSae,
    _YesNoLogitsProcessor,
    get_yes_no_token_ids,
    load_gemma3_model,
    resolve_target_module,
)
from gemma3_experiments.residual_sae.joint import apply_residual_feature_steering
from gemma3_experiments.residual_sae.models import load_residual_sae
from gemma3_experiments.residual_sae.normalization import load_residual_stats
from gemma3_experiments.residual_sae.devices import resolve_visible_device
from gemma3_experiments.residual_sae.datasets import build_simple_classify_prompt
from gemma3_experiments.residual_sae.config import GemmaResidualConfig
from gemma3_experiments.launch.residual.probe_validate import _read_jsonl

logger = logging.getLogger(__name__)

# Task → (task_key, pos_class, token_selection, token_role_for_patching)
TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "MAMI": {
        "task_key": "MAMI::misogyny_and_subtypes::label",
        "pos_class": "misogynous",
        "neg_class": "non-misogynous",
        "token_selection": "generated",
        "decode_strategy": "yes_no",
    },
    "HarMeme": {
        "task_key": "HarMeme::harmfulness_and_target::harmfulness",
        "pos_class": "harmful",
        "neg_class": "not harmful",
        "token_selection": "image",
        "decode_strategy": "yes_no",
    },
    "MMHS": {
        "task_key": "MMHS150K::hate_category::label",
        "pos_class": "Hate",
        "neg_class": "NotHate",
        "token_selection": "generated",
        "decode_strategy": "yes_no",
    },
    "Crisis_A": {
        "task_key": "Crisis_HateMM::subtask_a_hate_speech::label",
        "pos_class": "Hate Speech",
        "neg_class": "No Hate Speech",
        "token_selection": "image",
        "decode_strategy": "yes_no",
    },
}


@dataclass
class DynamicPatchingConfig:
    # Task: MAMI | HarMeme | MMHS | Crisis_A
    task_name: str = "MAMI"

    # 4B model (for MAMI / HarMeme / MMHS / Crisis_A)
    model_id: str = "google/gemma-3-4b-it"
    hook_module_path: str = "model.language_model.layers.22"

    # Base SAE (gemma-scope) loaded via GemmaSae.load_from_hub
    sae_repo_id: str = "google/gemma-scope-2-4b-it"
    sae_hookpoint: str = "resid_post/layer_22_width_65k_l0_medium"

    # Residual JumpReLU SAE checkpoint dir
    residual_sae_path: str = ""

    # Probe artifacts
    probe_joblib_dir: str = ""            # dir with probe_<task>.joblib files
    probe_val_predictions: str = ""       # probe_val_predictions.jsonl
    val_residual_cache_dir: str = ""      # residual SAE val cache (for source acts)
    top_k_probe_features: int = 20        # number of top probe features to ablate

    feature_source: str = "probe"         # probe | explicit
    explicit_feature_indices: list[int] = field(default_factory=list)
    # Tag written into the output filename/aggregate so silent vs routed runs don't clobber
    # each other, e.g. "silent" (default) or "routed".
    result_tag: str = "silent"

    # Experiment
    max_examples: int = 80               # max base-wrong/probe-right examples to test
    max_patch_sources: int = 30          # source examples to average over for patching
    output_dir: str = ""
    device: str = "cuda:0"


# ──────────────────────────────────────────────────────────────────────────────
# Load probe top features
# ──────────────────────────────────────────────────────────────────────────────

def _load_probe_top_features(
    joblib_dir: Path,
    task_key: str,
    top_k: int,
) -> tuple[list[int], list[float]]:
    """Return (feature_indices, weights) for top-K features by |weight| from probe."""
    task_safe = task_key.replace("::", "__").replace(" ", "_").replace("/", "_")
    candidates = list(joblib_dir.glob(f"probe_{task_safe}*.joblib"))
    if not candidates:
        # Try shorter name
        short = task_key.split("::")[0] + "__" + task_key.split("::")[1]
        candidates = list(joblib_dir.glob(f"probe_{short}*.joblib"))
    assert candidates, f"No probe joblib found for {task_key!r} in {joblib_dir}"
    probe_bundle = joblib.load(candidates[0])
    clf = probe_bundle["clf"]
    coef = clf.coef_
    if coef.ndim == 2 and coef.shape[0] == 1:
        coef = coef[0]  # binary LR stores [1, n_features]
    elif coef.ndim == 2:
        # Multiclass: use class with highest |weight|
        import numpy as np
        coef = coef[abs(coef).max(axis=1).argmax()]

    import numpy as np
    top_idx = (-abs(coef)).argsort()[:top_k]
    return [int(i) for i in top_idx], [float(coef[i]) for i in top_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Load source activations from residual cache
# ──────────────────────────────────────────────────────────────────────────────

def _load_source_activations(
    val_cache_dir: Path,
    sample_keys: list[str],
    top_feature_indices: list[int],
    residual_sae: Any,
    residual_stats: Any,
    device: str,
) -> dict[str, Any]:
    """Load per-sample mean feature activations for top K features.

    The residual cache stores dense residual error vectors [n_tokens, d_model].
    This function encodes them through the JumpReLU SAE and max-pools over tokens.
    Returns {sample_key: float32 array [K]}.
    """
    import numpy as np
    from safetensors import safe_open

    manifest = {
        f"{r['dataset_name']}::{r['task_name']}::{r['sample_id']}": r
        for r in _read_jsonl(val_cache_dir / "manifest.jsonl")
    }
    feat_idx_arr = top_feature_indices

    result: dict[str, Any] = {}
    shard_cache: dict[str, dict[str, Any]] = {}

    for key in sample_keys:
        row = manifest.get(key)
        if row is None:
            continue
        shard_path = str(val_cache_dir / row["shard_file"])
        if shard_path not in shard_cache:
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                shard_cache[shard_path] = {
                    k: f.get_tensor(k) for k in f.keys()
                }
        shard = shard_cache[shard_path]

        # Residual cache stores dense residual vectors [total_tokens, d_model]
        # offset/length index into the flat concatenated tokens for this sample
        residual_all = shard.get("residual")
        if residual_all is None:
            continue
        offset, length = int(row["offset"]), int(row["length"])
        residual_sample = residual_all[offset: offset + length]  # [n_tokens, d_model]

        # Optionally normalize before encoding
        residual_in = residual_sample.to(device).float()
        if residual_stats is not None:
            residual_in = residual_stats.normalize(residual_in)

        # Encode through JumpReLU SAE to get feature activations
        with torch.no_grad():
            if hasattr(residual_sae, 'encode'):
                acts = residual_sae.encode(residual_in)  # [n_tokens, n_latents]
                if isinstance(acts, tuple):
                    acts = acts[0]  # (values, indices) for TopK SAE
            else:
                acts = residual_sae.pre_acts(residual_in)

        # Max-pool over all tokens for the selected feature indices
        acts_cpu = acts.cpu().float()
        top_acts = acts_cpu[:, feat_idx_arr].max(dim=0).values.numpy()  # [K]
        result[key] = top_acts

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Margin measurement
# ──────────────────────────────────────────────────────────────────────────────

def _measure_yes_no_margin(
    model: Any,
    processor: Any,
    image: Any,
    prompt_text: str,
    yes_ids: list[int],
    no_ids: list[int],
    device: str,
) -> float:
    """Return logit(yes_token) - logit(no_token) at the first assistant token position."""
    from gemma3_experiments.residual_sae.gemma import build_multimodal_inputs
    inputs = build_multimodal_inputs(processor, image, prompt_text, device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    logits = outputs.logits[0, -1, :].float()  # last token logits [vocab]
    # yes = max logit among yes token variants; no = max among no variants
    yes_logit = max(logits[tid].item() for tid in yes_ids)
    no_logit  = max(logits[tid].item() for tid in no_ids)
    return yes_logit - no_logit


# ──────────────────────────────────────────────────────────────────────────────
# Make subspace hook
# ──────────────────────────────────────────────────────────────────────────────

def _make_subspace_hook(
    base_sae: GemmaSae,
    residual_sae: Any,
    residual_stats: Any,
    top_feature_indices: list[int],
    source_activations: list[float] | None,  # None = ablation; list = patch values
):
    """Return a forward hook that ablates (zeros) or patches (sets to source) the top K probe features."""

    def hook(_module: nn.Module, _inputs: tuple, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            if source_activations is None:
                # Ablation: zero the top K features
                values = [0.0] * len(top_feature_indices)
            else:
                values = source_activations
            updated, _ = apply_residual_feature_steering(
                hidden.float(),
                base_sae,
                residual_sae,
                feature_indices=top_feature_indices,
                feature_values=values,
                mode="clamp",
                residual_stats=residual_stats,
            )
            updated = updated.to(hidden.dtype)
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    return hook


# ──────────────────────────────────────────────────────────────────────────────
# Probe score computation from raw SAE activations
# ──────────────────────────────────────────────────────────────────────────────

def _probe_score_from_acts(
    acts: Any,  # [K] float array or None
    coef: Any,  # [n_features] full coef array
    top_feature_indices: list[int],
    intercept: float = 0.0,
) -> float | None:
    """Compute logistic regression score (before sigmoid) for the given acts."""
    if acts is None:
        return None
    import numpy as np
    score = float(coef[top_feature_indices] @ acts) + intercept
    return score


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: DynamicPatchingConfig) -> None:
    import numpy as np
    import random

    device = resolve_visible_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_meta = TASK_CONFIGS.get(cfg.task_name)
    assert task_meta, f"Unknown task: {cfg.task_name!r}. Choose from: {list(TASK_CONFIGS)}"
    task_key = task_meta["task_key"]
    pos_class = task_meta["pos_class"]
    neg_class = task_meta["neg_class"]
    logger.info("Task: %s  pos=%s  neg=%s", cfg.task_name, pos_class, neg_class)

    # ── Load probe features ────────────────────────────────────────────────────
    joblib_dir = Path(cfg.probe_joblib_dir)
    if cfg.feature_source == "explicit":
        assert cfg.explicit_feature_indices, (
            "feature_source=explicit requires --explicit_feature_indices "
            "(e.g. routing feature indices from analyze_residual_sae_routing_features.py)"
        )
        top_feat_idx = list(cfg.explicit_feature_indices)
        logger.info("Using %d EXPLICIT feature indices (feature_source=explicit): %s",
                    len(top_feat_idx), top_feat_idx[:5])
    else:
        top_feat_idx, top_feat_weights = _load_probe_top_features(
            joblib_dir, task_key, cfg.top_k_probe_features)
        logger.info("Top %d probe features: %s", len(top_feat_idx), top_feat_idx[:5])

    task_safe = task_key.replace("::", "__").replace(" ", "_").replace("/", "_")
    candidates = list(joblib_dir.glob(f"probe_{task_safe}*.joblib"))
    if not candidates:
        short = task_key.split("::")[0] + "__" + task_key.split("::")[1]
        candidates = list(joblib_dir.glob(f"probe_{short}*.joblib"))
    probe_bundle = joblib.load(candidates[0])
    clf = probe_bundle["clf"]
    # For multiclass (ndim==2, n_classes>1) take the class row with largest max |weight|,
    # matching the selection in _load_probe_top_features so full_coef is always 1-D.
    if clf.coef_.ndim == 2 and clf.coef_.shape[0] > 1:
        full_coef = clf.coef_[abs(clf.coef_).max(axis=1).argmax()]
    elif clf.coef_.ndim == 2:
        full_coef = clf.coef_[0]
    else:
        full_coef = clf.coef_
    probe_intercept = float(clf.intercept_[0]) if hasattr(clf, "intercept_") else 0.0

    # ── Identify base-wrong / probe-right examples ────────────────────────────
    all_preds = [r for r in _read_jsonl(Path(cfg.probe_val_predictions))
                 if r.get("task_key") == task_key]
    bw_pr = [r for r in all_preds
             if not r.get("baseline_correct") and r.get("probe_correct")]
    br_pr = [r for r in all_preds
             if r.get("baseline_correct") and r.get("probe_correct")]
    logger.info("base-wrong/probe-right: %d  base-right/probe-right: %d",
                len(bw_pr), len(br_pr))

    random.seed(42)
    targets = bw_pr[:cfg.max_examples]
    sources = random.sample(br_pr, min(cfg.max_patch_sources, len(br_pr)))
    logger.info("Using %d targets, %d sources", len(targets), len(sources))

    # ── Load model and SAEs (needed before residual cache encoding) ───────────
    logger.info("Loading 4B model and SAEs …")
    gcfg = GemmaResidualConfig()
    gcfg.model_id = cfg.model_id
    gcfg.hook_module_path = cfg.hook_module_path
    gcfg.sae_repo_id = cfg.sae_repo_id
    gcfg.sae_hookpoint = cfg.sae_hookpoint
    gcfg.model_device = device
    gcfg.sae_device = device
    model, processor = load_gemma3_model(gcfg)
    target_module = resolve_target_module(model, gcfg)
    base_sae = GemmaSae.load_from_hub(cfg.sae_repo_id, cfg.sae_hookpoint, device=device)
    residual_sae = load_residual_sae(Path(cfg.residual_sae_path), device=device)
    residual_stats = load_residual_stats(Path(cfg.residual_sae_path))
    yes_ids, no_ids = get_yes_no_token_ids(processor)
    logger.info("Model and SAEs loaded. yes_ids=%s no_ids=%s", yes_ids[:2], no_ids[:2])

    # ── Load + encode source / target activations from residual cache ─────────
    val_cache_dir = Path(cfg.val_residual_cache_dir)
    source_keys = [r["sample_key"] for r in sources]
    target_keys  = [r["sample_key"] for r in targets]

    logger.info("Encoding source activations from residual cache …")
    source_acts_by_key = _load_source_activations(
        val_cache_dir, source_keys, top_feat_idx,
        residual_sae=residual_sae, residual_stats=residual_stats, device=device)
    if source_acts_by_key:
        patch_mean_acts = np.stack(list(source_acts_by_key.values())).mean(0)
        logger.info("Loaded %d source activations, mean act: %.4f",
                    len(source_acts_by_key), float(patch_mean_acts.mean()))
    else:
        logger.warning("No source activations found — patching condition skipped")
        patch_mean_acts = None

    logger.info("Encoding target activations from residual cache …")
    target_acts_by_key = _load_source_activations(
        val_cache_dir, target_keys, top_feat_idx,
        residual_sae=residual_sae, residual_stats=residual_stats, device=device)

    # ── Load manifest for image paths and prompts ─────────────────────────────
    val_manifest = {
        f"{r['dataset_name']}::{r['task_name']}::{r['sample_id']}": r
        for r in _read_jsonl(val_cache_dir / "manifest.jsonl")
    }

    # ── Run experiment ────────────────────────────────────────────────────────
    results = []
    for i, target_row in enumerate(targets):
        sample_key = target_row["sample_key"]
        gold_label = target_row.get("gold_label", "")
        probe_pred = target_row.get("probe_pred", "")
        base_pred  = target_row.get("baseline_pred", "")

        manifest_row = val_manifest.get(sample_key)
        if manifest_row is None:
            logger.debug("No manifest entry for %s — skipping", sample_key)
            continue

        image_path = Path(manifest_row["image_path"])
        prompt_text = manifest_row.get("prompt_text", "")
        if not image_path.exists():
            logger.debug("Image not found: %s", image_path)
            continue
        image = Image.open(str(image_path)).convert("RGB")

        # Build simple classify prompt (yes/no style)
        try:
            simple_prompt = build_simple_classify_prompt(manifest_row, pos_class)
        except Exception:
            simple_prompt = prompt_text  # fallback

        if (i + 1) % 10 == 0:
            logger.info("  [%d/%d] %s", i + 1, len(targets), sample_key)

        # ── Baseline: no hook ─────────────────────────────────────────────────
        with torch.no_grad():
            baseline_margin = _measure_yes_no_margin(
                model, processor, image, simple_prompt, yes_ids, no_ids, device)

        # ── Ablation: zero top K probe features ───────────────────────────────
        ablation_hook_fn = _make_subspace_hook(
            base_sae, residual_sae, residual_stats,
            top_feat_idx, source_activations=None)
        handle = target_module.register_forward_hook(ablation_hook_fn)
        try:
            with torch.no_grad():
                ablation_margin = _measure_yes_no_margin(
                    model, processor, image, simple_prompt, yes_ids, no_ids, device)
        finally:
            handle.remove()

        # ── Patching: set top K features to source mean ───────────────────────
        patch_margin = None
        if patch_mean_acts is not None:
            patch_hook_fn = _make_subspace_hook(
                base_sae, residual_sae, residual_stats,
                top_feat_idx, source_activations=patch_mean_acts.tolist())
            handle = target_module.register_forward_hook(patch_hook_fn)
            try:
                with torch.no_grad():
                    patch_margin = _measure_yes_no_margin(
                        model, processor, image, simple_prompt, yes_ids, no_ids, device)
            finally:
                handle.remove()

        # ── Probe scores before/after ablation ────────────────────────────────
        target_acts_K = target_acts_by_key.get(sample_key)
        probe_score_before = _probe_score_from_acts(
            target_acts_K, full_coef, top_feat_idx, probe_intercept)
        # After ablation: zero the K features
        if target_acts_K is not None:
            acts_ablated = target_acts_K.copy()
            acts_ablated[:] = 0.0  # zeroed K features → all zeros
            probe_score_after = _probe_score_from_acts(
                acts_ablated, full_coef, top_feat_idx, probe_intercept)
        else:
            probe_score_after = None

        results.append({
            "sample_key": sample_key,
            "gold_label": gold_label,
            "probe_pred": probe_pred,
            "base_pred": base_pred,
            "baseline_margin": round(float(baseline_margin), 4),
            "ablation_margin": round(float(ablation_margin), 4),
            "patch_margin": round(float(patch_margin), 4) if patch_margin is not None else None,
            "delta_ablation": round(float(ablation_margin - baseline_margin), 4),
            "delta_patch": round(float(patch_margin - baseline_margin), 4) if patch_margin is not None else None,
            "probe_score_before": round(float(probe_score_before), 4) if probe_score_before is not None else None,
            "probe_score_after": round(float(probe_score_after), 4) if probe_score_after is not None else None,
            "probe_score_delta": round(float(probe_score_after - probe_score_before), 4)
                                 if (probe_score_before is not None and probe_score_after is not None) else None,
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    deltas_abl = [r["delta_ablation"] for r in results]
    deltas_pat = [r["delta_patch"] for r in results if r["delta_patch"] is not None]
    probe_deltas = [r["probe_score_delta"] for r in results if r["probe_score_delta"] is not None]

    aggregate = {
        "task_name": cfg.task_name,
        "task_key": task_key,
        "feature_source": cfg.feature_source,
        "result_tag": cfg.result_tag,
        "n_targets": len(results),
        "n_sources": len(sources),
        "top_k_features": len(top_feat_idx),
        "top_feature_indices": top_feat_idx,
        "ablation": {
            "n": len(deltas_abl),
            "mean_delta_margin": round(float(sum(deltas_abl) / len(deltas_abl)), 4) if deltas_abl else None,
            "mean_abs_delta_margin": round(float(sum(abs(d) for d in deltas_abl) / len(deltas_abl)), 4) if deltas_abl else None,
            "frac_margin_toward_correct": round(
                float(sum(1 for d in deltas_abl if d < 0) / len(deltas_abl)), 4
            ) if deltas_abl else None,
            # Hypothesis: ablation should make margin MORE negative (less likely correct)
            # For "base-wrong" examples the baseline margin is already negative
            # Ablation moves the margin further negative if probe features were helping
            "interpretation": (
                "frac_margin_toward_correct < 0.5 means ablating these features makes "
                "the base model even more wrong. For the SILENT (probe) feature set this "
                "supports silent knowledge (model ignores features the probe relies on). "
                "For the ROUTED (J-lens) feature set, a larger mean_abs_delta_margin than "
                "the silent condition -- regardless of direction -- supports the routed-but-"
                "generic hypothesis (the generative head is controllable through these "
                "features even though they may not be classification-discriminative)."
            ),
        },
        "patch": {
            "n": len(deltas_pat),
            "mean_delta_margin": round(float(sum(deltas_pat) / len(deltas_pat)), 4) if deltas_pat else None,
            "mean_abs_delta_margin": round(float(sum(abs(d) for d in deltas_pat) / len(deltas_pat)), 4) if deltas_pat else None,
            "frac_margin_toward_correct": round(
                float(sum(1 for d in deltas_pat if d > 0) / len(deltas_pat)), 4
            ) if deltas_pat else None,
        },
        "probe_score": {
            "n": len(probe_deltas),
            "mean_delta": round(float(sum(probe_deltas) / len(probe_deltas)), 4) if probe_deltas else None,
            "mean_abs_delta": round(float(sum(abs(d) for d in probe_deltas) / len(probe_deltas)), 4) if probe_deltas else None,
        },
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    task_safe = cfg.task_name.lower()
    (out_dir / f"silent_knowledge_{task_safe}_{cfg.result_tag}.json").write_text(
        json.dumps({"aggregate": aggregate, "per_sample": results}, indent=2))
    logger.info("Wrote results for %s/%s (%d examples)", cfg.task_name, cfg.result_tag, len(results))

    print(f"\n{'='*60}")
    print(f"Silent Knowledge: {cfg.task_name}  [{cfg.result_tag}]  (n={len(results)})")
    print(f"  Ablation:  mean_Δmargin={aggregate['ablation']['mean_delta_margin']}  "
          f"mean_|Δmargin|={aggregate['ablation']['mean_abs_delta_margin']}  "
          f"frac_toward_correct={aggregate['ablation']['frac_margin_toward_correct']}")
    print(f"  Patch:     mean_Δmargin={aggregate['patch']['mean_delta_margin']}  "
          f"frac_toward_correct={aggregate['patch']['frac_margin_toward_correct']}")
    print(f"  Probe Δscore: mean={aggregate['probe_score']['mean_delta']}")
    print(f"{'='*60}")
    print("Interpretation:")
    print("  If ablation frac_toward_correct ≈ 0.5 AND probe_score_delta is large:")
    print("  → Silent knowledge confirmed: probe features carry information the model ignores")
    print("  If ablation moves margin substantially toward correct/incorrect:")
    print("  → Probe features DO influence model output (not fully silent)")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(DynamicPatchingConfig,
                argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
