

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Task classification: which tasks use binary (yes/no) vs multiclass label_score
# ─────────────────────────────────────────────────────────────

BINARY_TASK_PATTERNS = [
    "hateful_memes_binary",
    "misogyny_and_subtypes",
    "offensive_detection",
    "subtask_a_hate_speech",
    "harm_binary",          # HarMeme binary collapsed task
]

# For multiclass tasks: maps a normalized class name to its representative label string
# used when tokenising for the logit-effect computation.
MULTICLASS_LABEL_MAP: dict[str, dict[str, str]] = {
    "hate_category": {
        "Racist": "Racist",
        "Sexist": "Sexist",
        "Homophobe": "Homophobe",
        "Religion": "Religion",
        "OtherHate": "Other",
        "NotHate": "No",
    },
    "harmfulness_and_target": {
        "not harmful": "No",
        "somewhat harmful": "Somewhat",
        "very harmful": "Very",
    },
    "subtask_b_hate_target": {
        "Community": "Community",
        "Individual": "Individual",
        "Organization": "Organization",
    },
}

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class FeatureLogitAlignmentConfig:
    # Probe top features from probe_sae_features.py or probe_fhm_crosscoder.py
    probe_top_features_path: str = ""

    # SAE checkpoint dir (for residual SAE) OR crosscoder HF repo
    sae_checkpoint_dir: str = ""          # residual SAE: contains sae.safetensors
    crosscoder_repo_id: str = ""          # e.g. "google/gemma-scope-2-4b-it"
    crosscoder_hookpoint: str = ""        # e.g. "crosscoder/layer_9_17_22_29_width_65k_l0_medium"
    crosscoder_layer_idx: int = 2         # which layer slice of W_dec to use (0=l9,1=l17,2=l22,3=l29)

    # Gemma model (for unembedding weights only — loaded on CPU)
    model_id: str = "google/gemma-3-4b-it"

    # Analysis settings
    top_k_features: int = 200             # how many probe-top features to analyse per class
    silent_threshold_logit: float = 0.1  # |logit_effect| < this → "silent"
    high_probe_percentile: float = 50.0  # abs(probe_weight) > this percentile → "high"

    # Task filter — leave empty to analyse all tasks in top_features.json
    tasks: list[str] = field(default_factory=list)

    output_dir: str = ""
    make_plots: bool = True


# ─────────────────────────────────────────────────────────────
# Weight loading helpers
# ─────────────────────────────────────────────────────────────

def _load_sae_w_dec(cfg: FeatureLogitAlignmentConfig) -> torch.Tensor:
    """Load decoder weight matrix.  Returns float32 tensor [num_latents, d_model]."""
    from safetensors.torch import load_file as safe_load

    if cfg.sae_checkpoint_dir:
        ckpt = Path(cfg.sae_checkpoint_dir)
        sae_file = ckpt / "sae.safetensors"
        assert sae_file.exists(), f"Missing {sae_file}"
        sd = safe_load(str(sae_file), device="cpu")
        for key in ("w_dec", "W_dec", "decoder.weight"):
            if key in sd:
                logger.info("Loaded residual SAE w_dec from %s  key=%s  shape=%s",
                            sae_file, key, tuple(sd[key].shape))
                return sd[key].float()
        raise KeyError(f"Cannot find w_dec in {sae_file}: keys={list(sd.keys())}")

    if cfg.crosscoder_repo_id and cfg.crosscoder_hookpoint:
        from huggingface_hub import snapshot_download

        repo_path = (
            Path(snapshot_download(
                cfg.crosscoder_repo_id,
                allow_patterns=[
                    f"{cfg.crosscoder_hookpoint}/*.safetensors",
                    f"{cfg.crosscoder_hookpoint}/*/*.safetensors",
                    f"{cfg.crosscoder_hookpoint}/*.safetensors.index.json",
                ],
            ))
            / cfg.crosscoder_hookpoint
        )
        safetensor_paths = sorted(repo_path.rglob("*.safetensors"))
        assert safetensor_paths, f"No safetensors found under {repo_path}"

        # For the crosscoder, each per-layer file contains the FULL w_dec
        # [num_latents, n_layers, d_model].  _load_multifile_state_dict would
        # stack them (producing spurious 4D tensor), so load from one file only.
        params_paths = [p for p in safetensor_paths if p.name == "params.safetensors"]
        single_path = params_paths[0] if params_paths else safetensor_paths[0]
        sd = safe_load(str(single_path), device="cpu")
        logger.info("Loading crosscoder w_dec from %s  (all keys: %s)", single_path.name, list(sd.keys()))

        w_dec: torch.Tensor | None = None
        for key in ("w_dec", "W_dec", "decoder.weight"):
            if key in sd:
                w_dec = sd[key].float()
                logger.info("Crosscoder w_dec: key=%s  shape=%s", key, tuple(w_dec.shape))
                break
        assert w_dec is not None, f"Cannot find w_dec in {single_path}: keys={list(sd.keys())}"

        if w_dec.ndim == 3:
            # [num_latents, n_layers, d_model] → slice requested layer
            w_dec_layer = w_dec[:, cfg.crosscoder_layer_idx, :]
            logger.info("Crosscoder w_dec[:, layer=%d, :]: shape=%s", cfg.crosscoder_layer_idx,
                        tuple(w_dec_layer.shape))
            return w_dec_layer
        if w_dec.ndim == 2:
            # Already [num_latents, d_model] (single-layer SAE used as crosscoder)
            return w_dec
        raise ValueError(f"Unexpected crosscoder w_dec ndim={w_dec.ndim}  shape={tuple(w_dec.shape)}")

    raise ValueError("Provide either --sae_checkpoint_dir or --crosscoder_repo_id + --crosscoder_hookpoint")


def _load_model_weights(model_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (lm_head_weight [vocab, d_model], norm_weight [d_model]) as float32 CPU tensors."""
    from transformers import Gemma3ForConditionalGeneration
    logger.info("Loading Gemma model weights (CPU, float32) from %s …", model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # LM head weight (tied to embed_tokens in Gemma)
    lm_head: torch.Tensor | None = None
    for attr_path in (
        "language_model.lm_head.weight",
        "lm_head.weight",
        "language_model.model.embed_tokens.weight",
    ):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            lm_head = obj.detach().float().cpu()
            logger.info("lm_head from %s  shape=%s", attr_path, tuple(lm_head.shape))
            break
        except AttributeError:
            continue
    assert lm_head is not None, "Cannot locate lm_head.weight on Gemma3ForConditionalGeneration"

    # Final layer norm weight
    norm_w: torch.Tensor | None = None
    for attr_path in (
        "language_model.model.norm.weight",
        "language_model.model.final_layernorm.weight",
        "model.norm.weight",
    ):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            norm_w = obj.detach().float().cpu()
            logger.info("final norm from %s  shape=%s", attr_path, tuple(norm_w.shape))
            break
        except AttributeError:
            continue

    del model
    return lm_head, norm_w  # norm_w may be None; handled downstream


def _build_unembed_eff(lm_head: torch.Tensor, norm_w: torch.Tensor | None) -> torch.Tensor:
    """Fold final RMSNorm weight into lm_head.  Returns W_U_eff [vocab, d_model]."""
    if norm_w is not None:
        return lm_head * norm_w[None, :]  # [vocab, d_model]
    return lm_head


def _yes_no_unembed_diff(
    W_U_eff: torch.Tensor,
    yes_ids: list[int],
    no_ids: list[int],
) -> torch.Tensor:
    """unembed_diff = max-col(W_U_eff[yes_ids]) - max-col(W_U_eff[no_ids])  → [d_model]."""
    pos_vecs = W_U_eff[torch.tensor(yes_ids)]   # [n_yes, d]
    neg_vecs = W_U_eff[torch.tensor(no_ids)]    # [n_no, d]
    return pos_vecs.max(0).values - neg_vecs.max(0).values  # [d]


def _label_unembed_diff(
    W_U_eff: torch.Tensor,
    pos_label: str,
    neg_labels: list[str],
    tokenizer: Any,
) -> torch.Tensor:
    """For multiclass: diff = W_U_eff[first_tok(pos_label)] - mean(W_U_eff[first_tok(neg)]).
    Returns shape [d_model].
    """
    def _first_tok(s: str) -> int:
        ids = tokenizer.encode(s, add_special_tokens=False)
        assert ids, f"Empty token IDs for label {s!r}"
        return ids[0]

    pos_vec = W_U_eff[_first_tok(pos_label)]                         # [d]
    neg_vecs = torch.stack([W_U_eff[_first_tok(l)] for l in neg_labels])  # [k, d]
    return pos_vec - neg_vecs.mean(0)                                  # [d]


# ─────────────────────────────────────────────────────────────
# Task helpers
# ─────────────────────────────────────────────────────────────

def _is_binary_task(task_key: str) -> bool:
    task_lower = task_key.lower()
    return any(p in task_lower for p in BINARY_TASK_PATTERNS)


def _get_multiclass_labels(task_key: str) -> dict[str, str] | None:
    """Return {class_name: label_string_to_tokenize} for the task or None if not found."""
    for pattern, mapping in MULTICLASS_LABEL_MAP.items():
        if pattern in task_key.lower():
            return mapping
    return None


# ─────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────

def _classify_quadrant(probe_w: float, logit_eff: float, silent_thr: float) -> str:
    if abs(logit_eff) < silent_thr:
        return "silent" if abs(probe_w) > 0 else "null"
    if (probe_w > 0) == (logit_eff > 0):
        return "aligned"
    return "misaligned"


def _analyse_task(
    task_key: str,
    class_features: dict[str, list[dict]],  # class_name → [{feature_idx, weight}]
    w_dec: torch.Tensor,                     # [num_latents, d_model]
    W_U_eff: torch.Tensor,                   # [vocab, d_model]
    yes_ids: list[int],
    no_ids: list[int],
    tokenizer: Any,
    cfg: FeatureLogitAlignmentConfig,
    top_k: int,
) -> dict[str, Any]:
    results_by_class: dict[str, list[dict]] = {}
    summary_by_class: dict[str, dict] = {}

    # Determine whether binary or multiclass
    is_binary = _is_binary_task(task_key)
    mc_labels = None if is_binary else _get_multiclass_labels(task_key)

    for class_name, feat_list in class_features.items():
        feat_list_k = feat_list[:top_k]
        if not feat_list_k:
            continue

        # Build unembedding diff for this class
        if is_binary:
            # For binary tasks with constrained decode: yes=positive, no=negative.
            # The probe's class_name may be "hateful", "misogynous" etc. — these map to yes.
            if "not" in class_name.lower() or "non" in class_name.lower() or class_name.lower() in ("not hateful", "non-misogynous", "non-offensiv", "no hate speech", "not harmful"):
                unembed_diff = _yes_no_unembed_diff(W_U_eff, no_ids, yes_ids)  # flipped
            else:
                unembed_diff = _yes_no_unembed_diff(W_U_eff, yes_ids, no_ids)
        elif mc_labels is not None:
            label_str = mc_labels.get(class_name)
            if label_str is None:
                # Try case-insensitive fallback
                label_str = next(
                    (v for k, v in mc_labels.items() if k.lower() == class_name.lower()), None
                )
            if label_str is None:
                logger.warning("No label string for class %r in task %s; skipping.", class_name, task_key)
                continue
            neg_labels = [v for k, v in mc_labels.items() if k.lower() != class_name.lower()]
            try:
                unembed_diff = _label_unembed_diff(W_U_eff, label_str, neg_labels, tokenizer)
            except Exception as exc:
                logger.warning("unembed_diff failed for %s / %s: %s", task_key, class_name, exc)
                continue
        else:
            # Unknown task type: fall back to binary yes/no
            logger.warning("Unknown multiclass task %s; falling back to yes/no.", task_key)
            unembed_diff = _yes_no_unembed_diff(W_U_eff, yes_ids, no_ids)

        # Compute logit effects for all top features at once via batch matmul
        feat_indices = [int(f["feature_idx"]) for f in feat_list_k]
        probe_weights = torch.tensor([float(f["weight"]) for f in feat_list_k], dtype=torch.float32)
        w_dec_sub = w_dec[torch.tensor(feat_indices)]           # [k, d_model]
        logit_effects = (w_dec_sub @ unembed_diff).float()      # [k]

        # Probe weight scale (for percentile threshold)
        abs_pw = probe_weights.abs()
        pw_thresh = float(torch.quantile(abs_pw, cfg.high_probe_percentile / 100.0))

        rows = []
        for idx, feat_idx in enumerate(feat_indices):
            pw = float(probe_weights[idx])
            le = float(logit_effects[idx])
            is_high_probe = abs(pw) >= pw_thresh
            quad = _classify_quadrant(pw, le, cfg.silent_threshold_logit)
            rows.append({
                "feature_idx": feat_idx,
                "probe_weight": round(pw, 6),
                "static_logit_effect": round(le, 6),
                "quadrant": quad,
                "high_probe": is_high_probe,
            })

        results_by_class[class_name] = rows

        # Summary
        quad_counts: dict[str, int] = {}
        for r in rows:
            quad_counts[r["quadrant"]] = quad_counts.get(r["quadrant"], 0) + 1

        # Top silent features (high probe weight, near-zero logit effect)
        silent = [r for r in rows if r["high_probe"] and r["quadrant"] == "silent"]
        silent.sort(key=lambda r: -abs(r["probe_weight"]))

        # Top aligned
        aligned = [r for r in rows if r["high_probe"] and r["quadrant"] == "aligned"]
        aligned.sort(key=lambda r: -(abs(r["probe_weight"]) * abs(r["static_logit_effect"])))

        summary_by_class[class_name] = {
            "n_features": len(rows),
            "probe_weight_scale": round(float(abs_pw.mean()), 4),
            "logit_effect_scale": round(float(logit_effects.abs().mean()), 4),
            "quadrant_counts": quad_counts,
            "top_silent_features": silent[:10],
            "top_aligned_features": aligned[:10],
            "silent_rate": round(sum(1 for r in rows if r["high_probe"] and r["quadrant"] == "silent") / max(sum(1 for r in rows if r["high_probe"]), 1), 4),
        }

    return {"features": results_by_class, "summary": summary_by_class}


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def _make_scatter(
    rows: list[dict],
    task_key: str,
    class_name: str,
    output_dir: Path,
    silent_thr: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping scatter plots")
        return

    pws = [r["probe_weight"] for r in rows]
    les = [r["static_logit_effect"] for r in rows]
    quads = [r["quadrant"] for r in rows]
    colors = {"aligned": "steelblue", "silent": "orange", "misaligned": "red", "null": "gray"}
    c = [colors.get(q, "gray") for q in quads]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pws, les, c=c, alpha=0.5, s=20)
    ax.axhline(silent_thr, color="orange", linewidth=0.8, linestyle="--", label=f"silent thr ±{silent_thr}")
    ax.axhline(-silent_thr, color="orange", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Probe weight")
    ax.set_ylabel("Static logit effect (W_dec · unembed_diff)")
    slug = re.sub(r"[^\w]", "_", f"{task_key}_{class_name}")[:80]
    ax.set_title(f"{task_key}\n{class_name}", fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in colors.items()]
    ax.legend(handles=legend_elements, fontsize=7)

    out_path = output_dir / f"scatter_{slug}.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    logger.info("Saved scatter: %s", out_path)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run(cfg: FeatureLogitAlignmentConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load top features ──────────────────────────────────────────────────────
    assert cfg.probe_top_features_path, "--probe_top_features_path required"
    top_features: dict[str, dict[str, list[dict]]] = json.loads(
        Path(cfg.probe_top_features_path).read_text()
    )
    # Normalise to task_key → class_name → [{feature_idx, weight, abs_weight}]
    # probe_sae_features.py saves:  {task_key: {class_name: [...]}}
    # probe_fhm_crosscoder.py saves: {class_name: [...]}  (single task)
    if not isinstance(next(iter(top_features.values())), dict):
        # crosscoder format: wrap in a single task
        task_name = Path(cfg.probe_top_features_path).parent.name
        top_features = {task_name: top_features}

    if cfg.tasks:
        top_features = {k: v for k, v in top_features.items() if k in cfg.tasks}

    logger.info("Analysing %d tasks: %s", len(top_features), list(top_features))

    # ── Load SAE W_dec ─────────────────────────────────────────────────────────
    w_dec = _load_sae_w_dec(cfg)   # [num_latents, d_model]
    num_latents, d_model = w_dec.shape
    logger.info("w_dec: num_latents=%d d_model=%d", num_latents, d_model)

    # ── Load model weights ─────────────────────────────────────────────────────
    lm_head, norm_w = _load_model_weights(cfg.model_id)
    W_U_eff = _build_unembed_eff(lm_head, norm_w)   # [vocab, d_model]
    logger.info("W_U_eff: %s", tuple(W_U_eff.shape))

    # ── Get yes/no token IDs ───────────────────────────────────────────────────
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    from gemma3_experiments.residual_sae.gemma import get_yes_no_token_ids
    yes_ids, no_ids = get_yes_no_token_ids(processor)
    logger.info("yes_ids=%s  no_ids=%s", yes_ids, no_ids)
    tokenizer = processor.tokenizer

    # ── Analyse each task ──────────────────────────────────────────────────────
    all_results: dict[str, Any] = {}
    all_summaries: dict[str, Any] = {}

    for task_key, class_features in top_features.items():
        logger.info("Task: %s  classes: %s", task_key, list(class_features))
        result = _analyse_task(
            task_key=task_key,
            class_features=class_features,
            w_dec=w_dec,
            W_U_eff=W_U_eff,
            yes_ids=yes_ids,
            no_ids=no_ids,
            tokenizer=tokenizer,
            cfg=cfg,
            top_k=cfg.top_k_features,
        )
        all_results[task_key] = result["features"]
        all_summaries[task_key] = result["summary"]

        if cfg.make_plots:
            for class_name, rows in result["features"].items():
                _make_scatter(rows, task_key, class_name, output_dir, cfg.silent_threshold_logit)

    # ── Save outputs ───────────────────────────────────────────────────────────
    (output_dir / "feature_logit_alignment.json").write_text(
        json.dumps(all_results, indent=2)
    )
    (output_dir / "feature_logit_alignment_summary.json").write_text(
        json.dumps(all_summaries, indent=2)
    )

    # Print human-readable summary
    for task_key, task_sum in all_summaries.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_key}")
        for cls, s in task_sum.items():
            print(f"  [{cls}]  n={s['n_features']}  pw_scale={s['probe_weight_scale']}  "
                  f"le_scale={s['logit_effect_scale']}  silent_rate={s['silent_rate']}")
            print(f"    quadrants: {s['quadrant_counts']}")
            if s.get("top_silent_features"):
                feats = [f"f{r['feature_idx']}(pw={r['probe_weight']:.3f},le={r['static_logit_effect']:.3f})"
                         for r in s["top_silent_features"][:5]]
                print(f"    top silent: {', '.join(feats)}")

    logger.info("Wrote outputs to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FeatureLogitAlignmentConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
