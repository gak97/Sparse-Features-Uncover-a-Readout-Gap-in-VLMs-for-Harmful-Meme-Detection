

import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from simple_parsing import parse

from gemma3_experiments.residual_sae.datasets import (
    build_simple_classify_prompt,
)
from gemma3_experiments.residual_sae.normalization import load_residual_stats

from ..config import QwenInferenceConfig
from ..joint import make_joint_reconstruction_hook
from ..models import load_residual_sae
from ..qwen import (
    QwenSae,
    build_multimodal_inputs,
    classify_constrained,
    get_yes_no_token_ids,
    load_qwen3_model,
    resolve_target_module,
    score_label_candidates,
)

logger = logging.getLogger(__name__)

# Import dataset utilities from gemma3 (shared across both pipelines).
from gemma3_experiments.residual_sae.datasets import (
    MemeSample,
    load_meme_samples,
)

try:
    from gemma3_experiments.residual_sae.datasets import build_primary_label_prompt
except ImportError:
    def build_primary_label_prompt(sample: MemeSample) -> str:  # type: ignore[misc]
        labels = " / ".join(sample.schema.primary_field.labels)
        return (
            f"{sample.schema.task_description}\n"
            f"The meme text is: \"{sample.meme_text}\"\n"
            f"Choose exactly one label: {labels}\nAnswer:"
        )


def _is_binary_task(sample: MemeSample) -> bool:
    return len(sample.schema.primary_field.labels) == 2


def _resolve_checkpoint(requested: str) -> Path:
    path = Path(requested)
    if not path.exists():
        raise FileNotFoundError(f"Residual SAE checkpoint not found: {path}")
    # Quick finite-weight check.
    weights = path / "sae.safetensors"
    if weights.exists():
        tensors = load_file(str(weights))
        if not all(torch.isfinite(t).all() for t in tensors.values()):
            # Try sibling checkpoints.
            for sibling in sorted(path.parent.iterdir()):
                sw = sibling / "sae.safetensors"
                if sw.exists() and sibling != path:
                    st = load_file(str(sw))
                    if all(torch.isfinite(t).all() for t in st.values()):
                        logger.warning("Checkpoint %s has non-finite weights; using %s", path, sibling)
                        return sibling
            raise RuntimeError(f"Checkpoint {path} has non-finite weights and no finite fallback found.")
    return path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(QwenInferenceConfig)

    samples = load_meme_samples(cfg.dataset)
    if cfg.binary_only:
        before = len(samples)
        samples = [s for s in samples if _is_binary_task(s)]
        logger.info("Binary-only: kept %d / %d samples.", len(samples), before)
    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]

    residual_sae_path: Path | None = None
    if cfg.apply_residual_sae:
        residual_sae_path = _resolve_checkpoint(cfg.residual_sae_path)

    logger.info(
        "Joint inference: n=%d apply_sae=%s alpha=%.3f decode=%s binary_only=%s output=%s",
        len(samples),
        cfg.apply_residual_sae,
        cfg.reconstruction_alpha,
        cfg.decode_strategy,
        cfg.binary_only,
        cfg.output_path,
    )

    model, processor = load_qwen3_model(cfg.qwen)
    target_module = resolve_target_module(model, cfg.qwen)

    hook = None
    if cfg.apply_residual_sae:
        assert residual_sae_path is not None
        base_sae = QwenSae.load_from_hub(
            cfg.qwen.sae_repo_id,
            cfg.qwen.sae_layer,
            device=cfg.qwen.model_device,
        )
        residual_sae = load_residual_sae(residual_sae_path, device=cfg.qwen.model_device)
        residual_stats = load_residual_stats(residual_sae_path)

        # Resolve feature indices and deltas for steering mode
        feature_indices = None
        feature_deltas = None
        if cfg.steering_feature_indices:
            feature_indices = cfg.steering_feature_indices
            feature_deltas = cfg.steering_deltas or [cfg.steering_delta] * len(feature_indices)
        elif cfg.steering_feature_idx is not None:
            feature_indices = [cfg.steering_feature_idx]
            feature_deltas = [cfg.steering_delta]

        hook = make_joint_reconstruction_hook(
            base_sae,
            residual_sae,
            feature_indices=feature_indices,
            feature_deltas=feature_deltas,
            steering_mode=cfg.steering_mode,
            reconstruction_alpha=cfg.reconstruction_alpha,
            residual_stats=residual_stats,
        )
        logger.info(
            "SAEs loaded: base=%s layer=%d residual=%s stats=%s steering=%s",
            cfg.qwen.sae_repo_id, cfg.qwen.sae_layer, residual_sae_path,
            "on" if residual_stats is not None else "off",
            f"features={feature_indices} deltas={feature_deltas}" if feature_indices else "reconstruction_only",
        )
    else:
        logger.info("Running baseline inference (no hook).")

    yes_ids, no_ids = get_yes_no_token_ids(processor)
    logger.info("yes_ids=%s no_ids=%s", yes_ids, no_ids)

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    log_every = max(1, len(samples) // 20)

    with output_path.open("w") as fd:
        for sample_i, sample in enumerate(samples, start=1):
            image = Image.open(sample.image_path).convert("RGB")

            handle = target_module.register_forward_hook(hook) if hook is not None else None
            try:
                # Determine decode strategy for this sample.
                strategy = cfg.decode_strategy
                if strategy == "auto":
                    strategy = "yes_no" if _is_binary_task(sample) else "label_score"

                if strategy == "yes_no":
                    prompt_text = build_simple_classify_prompt(sample)
                    is_positive, prediction_text, conf_gap = classify_constrained(
                        model, processor, image, prompt_text, cfg.qwen.model_device
                    )
                    primary_labels = sample.schema.primary_field.labels
                    pred_label = primary_labels[-1] if is_positive else primary_labels[0]
                    row = {
                        **sample.to_manifest_dict(),
                        "prompt_text": prompt_text,
                        "prediction_text": prediction_text,
                        "prediction_label": pred_label,
                        "is_positive": is_positive,
                        "conf_gap": round(conf_gap, 4),
                        "decode_strategy": "yes_no",
                        "apply_residual_sae": cfg.apply_residual_sae,
                        "reconstruction_alpha": cfg.reconstruction_alpha if cfg.apply_residual_sae else 0.0,
                        "residual_sae_path": str(residual_sae_path) if residual_sae_path else None,
                    }
                else:  # label_score
                    prompt_text = build_primary_label_prompt(sample)
                    label_texts = list(sample.schema.primary_field.labels)
                    pred_label, label_scores, confidence = score_label_candidates(
                        model, processor, image, prompt_text, label_texts, cfg.qwen.model_device
                    )
                    row = {
                        **sample.to_manifest_dict(),
                        "prompt_text": prompt_text,
                        "prediction_text": pred_label,
                        "prediction_label": pred_label,
                        "candidate_labels": label_texts,
                        "label_scores": {k: round(v, 4) for k, v in label_scores.items()},
                        "prediction_confidence": round(confidence, 4),
                        "decode_strategy": "label_score",
                        "apply_residual_sae": cfg.apply_residual_sae,
                        "reconstruction_alpha": cfg.reconstruction_alpha if cfg.apply_residual_sae else 0.0,
                        "residual_sae_path": str(residual_sae_path) if residual_sae_path else None,
                    }
            finally:
                if handle is not None:
                    handle.remove()

            fd.write(json.dumps(row) + "\n")

            if sample_i == 1 or sample_i % log_every == 0 or sample_i == len(samples):
                elapsed = time.monotonic() - t_start
                rate = sample_i / elapsed if elapsed > 0 else 0.0
                eta = (len(samples) - sample_i) / rate if rate > 0 else float("inf")
                pred_info = f"pred={row.get('prediction_text', '?')!r}"
                if "conf_gap" in row:
                    pred_info += f" conf={row['conf_gap']:.3f}"
                logger.info(
                    "%d/%d (%.1f%%) dataset=%s id=%s strategy=%s %s "
                    "elapsed=%.0fs rate=%.2f/s ETA=%.0fs",
                    sample_i, len(samples), 100.0 * sample_i / len(samples),
                    sample.dataset_name, sample.sample_id,
                    row["decode_strategy"], pred_info,
                    elapsed, rate, eta,
                )

    logger.info("Done: %d samples → %s in %.0fs", len(samples), output_path, time.monotonic() - t_start)


if __name__ == "__main__":
    main()
