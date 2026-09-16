

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from simple_parsing import parse, Serializable

logger = logging.getLogger(__name__)

BINARY_TASK_PATTERNS = [
    "hateful_memes_binary",
    "misogyny_and_subtypes",
    "offensive_detection",
    "subtask_a_hate_speech",
]


@dataclass
class FeatureLogitAlignmentConfig(Serializable):
    # Path to top_features.json from probe_sae_features.py (residual SAE) or
    # probe_base_sae.py (base SAE) — must match --sae_source below.
    probe_top_features_path: str = ""
    # Residual SAE checkpoint directory (contains sae.safetensors). Only used
    # when sae_source=residual.
    sae_checkpoint_dir: str = ""
    # Which SAE's decoder to align against. "base" is now the primary/mandatory
    # path per review feedback (base SAE outperforms residual SAE on every
    # locked-protocol task); "residual" is kept for the ablation comparison.
    sae_source: str = "base"        # base | residual
    # Base SAE (public Qwen release). Only used when sae_source=base.
    base_sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    base_sae_layer: int = 20
    # Qwen model id (loaded CPU-only for weight extraction)
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    # Analysis settings
    top_k_features: int = 200
    silent_threshold_logit: float = 0.1   # |logit_effect| < this → "silent"
    high_probe_percentile: float = 50.0   # abs(probe_weight) > this pctile → "high"
    make_plots: bool = True
    output_dir: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_sae_w_dec(checkpoint_dir: str) -> torch.Tensor:
    """Load residual SAE decoder matrix.  Returns float32 [num_latents, d_model]."""
    from safetensors.torch import load_file as safe_load

    ckpt = Path(checkpoint_dir)
    sae_file = ckpt / "sae.safetensors"
    assert sae_file.exists(), f"Missing SAE file: {sae_file}"
    sd = safe_load(str(sae_file), device="cpu")
    # JumpReLU SAE stores W_dec as parameter key "W_dec" shape [d_sae, d_model]
    for key in ("W_dec", "w_dec", "decoder.weight"):
        if key in sd:
            w = sd[key].float()
            logger.info("Loaded w_dec from %s  key=%s  shape=%s", sae_file, key, tuple(w.shape))
            return w
    raise KeyError(f"Cannot find w_dec in {sae_file}: keys={list(sd.keys())}")


def _load_base_sae_w_dec(repo_id: str, layer: int) -> torch.Tensor:
    """Load the PUBLIC base Qwen TopK SAE decoder matrix.

    QwenSae stores W_dec as [d_model, d_sae] (see qwen3_residual_sae/qwen.py);
    transpose to [d_sae, d_model] to match the residual-SAE convention used by
    `_compute_alignment` (indexed as `w_dec[feature_idx]` -> [d_model]).
    """
    from ..qwen import QwenSae

    base_sae = QwenSae.load_from_hub(repo_id, layer, device="cpu")
    w_dec = base_sae.W_dec.detach().float().T.contiguous()  # [d_sae, d_model]
    logger.info("Loaded base SAE w_dec from %s layer %d  shape=%s", repo_id, layer, tuple(w_dec.shape))
    return w_dec


def _load_qwen_lm_weights(model_id: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load (lm_head_weight [vocab, d_model], norm_weight [d_model]) as float32 CPU.

    Loads the model in bfloat16 to minimise RAM (~18 GB for 9B), extracts
    the two needed tensors as float32, then deletes the model.
    """
    from transformers import Qwen3_5ForConditionalGeneration

    logger.info("Loading Qwen3.5 model weights (CPU, bfloat16) from %s …", model_id)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    lm_head: torch.Tensor | None = None
    for attr_path in (
        "language_model.lm_head.weight",
        "language_model.model.embed_tokens.weight",   # tied weights fallback
        "lm_head.weight",
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
    assert lm_head is not None, "Cannot locate lm_head.weight on Qwen3_5ForConditionalGeneration"

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
    gc.collect()
    return lm_head, norm_w


def _build_unembed_eff(lm_head: torch.Tensor, norm_w: torch.Tensor | None) -> torch.Tensor:
    """Fold final RMSNorm weight into lm_head.  Returns W_U_eff [vocab, d_model]."""
    if norm_w is not None:
        return lm_head * norm_w[None, :]
    return lm_head


def _get_yes_no_ids(model_id: str) -> tuple[list[int], list[int]]:
    """Return single-token yes/no IDs from the Qwen tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    yes_ids, no_ids = [], []
    for text in (" Yes", " yes", "Yes", "yes"):
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in yes_ids:
            yes_ids.append(ids[0])
    for text in (" No", " no", "No", "no"):
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in no_ids:
            no_ids.append(ids[0])
    logger.info("yes_ids=%s  no_ids=%s", yes_ids, no_ids)
    assert yes_ids and no_ids, "Could not resolve single-token yes/no IDs"
    return yes_ids, no_ids


def _yes_no_unembed_diff(
    W_U_eff: torch.Tensor,
    yes_ids: list[int],
    no_ids: list[int],
) -> torch.Tensor:
    """unembed_diff = max-col(W_U_eff[yes_ids]) - max-col(W_U_eff[no_ids])  → [d_model]."""
    pos_vecs = W_U_eff[torch.tensor(yes_ids)]   # [n_yes, d]
    neg_vecs = W_U_eff[torch.tensor(no_ids)]    # [n_no,  d]
    return pos_vecs.max(0).values - neg_vecs.max(0).values


# ─────────────────────────────────────────────────────────────────────────────
# Core alignment computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_alignment(
    feat_indices: list[int],
    feat_weights: list[float],
    w_dec: torch.Tensor,         # [num_latents, d_model]
    unembed_diff: torch.Tensor,  # [d_model]
    silent_threshold: float,
    high_probe_percentile: float,
) -> list[dict]:
    """Compute logit alignment for a list of (feature_idx, probe_weight) pairs."""
    unembed_diff = unembed_diff.float()
    records = []
    abs_weights = [abs(w) for w in feat_weights]
    threshold_high = float(np.percentile(abs_weights, high_probe_percentile)) if abs_weights else 0.0

    for feat_idx, probe_w in zip(feat_indices, feat_weights):
        dec_dir = w_dec[feat_idx].float()  # [d_model]
        logit_effect = float(torch.dot(dec_dir, unembed_diff).item())
        abs_probe = abs(probe_w)
        is_high_probe = abs_probe > threshold_high
        is_silent = abs(logit_effect) < silent_threshold
        is_aligned = (probe_w > 0 and logit_effect > 0) or (probe_w < 0 and logit_effect < 0)

        if is_high_probe and is_silent:
            quadrant = "silent"
        elif is_high_probe and is_aligned:
            quadrant = "aligned"
        elif is_high_probe and not is_aligned:
            quadrant = "misaligned"
        else:
            quadrant = "low_probe"

        records.append({
            "feature_idx": feat_idx,
            "probe_weight": round(float(probe_w), 6),
            "logit_effect": round(float(logit_effect), 6),
            "quadrant": quadrant,
        })
    return records


def run(cfg: FeatureLogitAlignmentConfig) -> None:
    if cfg.output_dir:
        out_dir = Path(cfg.output_dir)
    elif cfg.sae_source == "residual":
        out_dir = Path(cfg.sae_checkpoint_dir) / "alignment"
    else:
        out_dir = Path("./qwen_base_sae_feature_logit_alignment")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ────────────────────────────────────────────────────────────
    top_feats_path = Path(cfg.probe_top_features_path)
    assert top_feats_path.exists(), f"Missing top_features.json: {top_feats_path}"
    top_features: dict[str, dict] = json.loads(top_feats_path.read_text())

    w_dec = (
        _load_base_sae_w_dec(cfg.base_sae_repo_id, cfg.base_sae_layer)
        if cfg.sae_source == "base"
        else _load_sae_w_dec(cfg.sae_checkpoint_dir)
    )   # [d_sae, d_model]
    lm_head, norm_w = _load_qwen_lm_weights(cfg.model_id)
    W_U_eff = _build_unembed_eff(lm_head, norm_w)     # [vocab, d_model]
    logger.info("W_U_eff shape: %s", tuple(W_U_eff.shape))

    yes_ids, no_ids = _get_yes_no_ids(cfg.model_id)
    unembed_diff = _yes_no_unembed_diff(W_U_eff, yes_ids, no_ids)  # [d_model]
    logger.info("unembed_diff norm: %.4f", float(unembed_diff.norm().item()))

    # ── Per-task alignment ─────────────────────────────────────────────────────
    all_alignment: dict[str, dict] = {}
    summary: dict[str, Any] = {}

    for task_key, directions in sorted(top_features.items()):
        logger.info("Processing task: %s", task_key)
        task_result: dict[str, Any] = {}

        for direction in ("label_up", "label_down"):
            entries = directions.get(direction, [])[:cfg.top_k_features]
            if not entries:
                continue
            feat_indices = [e["feature_idx"] for e in entries]
            feat_weights = [e["weight"] for e in entries]

            # For label_down, negate unembed_diff (negative label direction)
            ud = unembed_diff if direction == "label_up" else -unembed_diff

            records = _compute_alignment(
                feat_indices=feat_indices,
                feat_weights=feat_weights,
                w_dec=w_dec,
                unembed_diff=ud,
                silent_threshold=cfg.silent_threshold_logit,
                high_probe_percentile=cfg.high_probe_percentile,
            )
            task_result[direction] = records

        all_alignment[task_key] = task_result

        # Quadrant counts for summary
        all_records = [r for d in task_result.values() for r in d]
        counts = {}
        for r in all_records:
            counts[r["quadrant"]] = counts.get(r["quadrant"], 0) + 1
        silent_feats = [r for d in task_result.values() for r in d if r["quadrant"] == "silent"]
        top_silent = sorted(silent_feats, key=lambda x: -abs(x["probe_weight"]))[:10]
        summary[task_key] = {
            "quadrant_counts": counts,
            "n_total": len(all_records),
            "silent_fraction": counts.get("silent", 0) / max(1, len(all_records)),
            "top_silent_features": top_silent,
        }
        logger.info(
            "  %s — quadrants: %s  silent_fraction=%.3f",
            task_key, counts, summary[task_key]["silent_fraction"],
        )

    # ── Write outputs ─────────────────────────────────────────────────────────
    align_path = out_dir / "feature_logit_alignment.json"
    with align_path.open("w") as fd:
        json.dump(all_alignment, fd, indent=2)
    logger.info("Alignment written to %s", align_path)

    summary_path = out_dir / "feature_logit_alignment_summary.json"
    with summary_path.open("w") as fd:
        json.dump(summary, fd, indent=2)
    logger.info("Summary written to %s", summary_path)

    # ── Scatter plots ─────────────────────────────────────────────────────────
    if cfg.make_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for task_key, task_result in all_alignment.items():
                all_records = [r for d in task_result.values() for r in d]
                if not all_records:
                    continue

                probe_ws = np.array([r["probe_weight"] for r in all_records])
                logit_es = np.array([r["logit_effect"] for r in all_records])
                quadrants = [r["quadrant"] for r in all_records]
                colors = {
                    "aligned": "tab:blue",
                    "silent": "tab:orange",
                    "misaligned": "tab:red",
                    "low_probe": "tab:gray",
                }
                c = [colors.get(q, "black") for q in quadrants]

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(probe_ws, logit_es, c=c, alpha=0.5, s=12)
                ax.axhline(0, color="k", lw=0.5)
                ax.axvline(0, color="k", lw=0.5)
                ax.axhline(cfg.silent_threshold_logit, color="gray", lw=0.5, ls="--")
                ax.axhline(-cfg.silent_threshold_logit, color="gray", lw=0.5, ls="--")
                ax.set_xlabel("Probe weight")
                ax.set_ylabel("Logit effect (yes−no)")
                task_label = task_key.replace("::", " / ")
                ax.set_title(f"Feature→Logit Alignment\n{task_label}")
                for q, col in colors.items():
                    n = sum(1 for x in quadrants if x == q)
                    ax.scatter([], [], c=col, label=f"{q} ({n})", s=12)
                ax.legend(fontsize=7, loc="lower right")
                fig.tight_layout()
                safe_key = task_key.replace("::", "__").replace("/", "_")
                fig.savefig(out_dir / f"scatter_{safe_key}.png", dpi=120)
                plt.close(fig)

            logger.info("Scatter plots saved to %s", out_dir)
        except ImportError:
            logger.warning("matplotlib not available; skipping plots")

    # Print summary table
    print(f"\n{'Task':<55} {'silent%':>8} {'aligned%':>9} {'misalignd%':>11}")
    print("-" * 85)
    for task_key, s in sorted(summary.items()):
        n = s["n_total"]
        silent_pct = 100 * s["quadrant_counts"].get("silent", 0) / max(1, n)
        aligned_pct = 100 * s["quadrant_counts"].get("aligned", 0) / max(1, n)
        misalign_pct = 100 * s["quadrant_counts"].get("misaligned", 0) / max(1, n)
        print(f"  {task_key:<53} {silent_pct:>7.1f}% {aligned_pct:>8.1f}% {misalign_pct:>10.1f}%")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FeatureLogitAlignmentConfig)
    assert cfg.probe_top_features_path, "--probe_top_features_path is required"
    if cfg.sae_source == "residual":
        assert cfg.sae_checkpoint_dir, "--sae_checkpoint_dir is required when --sae_source=residual"
    run(cfg)


if __name__ == "__main__":
    main()
