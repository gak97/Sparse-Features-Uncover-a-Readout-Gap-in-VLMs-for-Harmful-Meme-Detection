import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from simple_parsing import parse


from gemma3_experiments.residual_sae import (
    ResidualInferenceConfig,
    build_classification_prompt,
    build_primary_label_prompt,
    build_multimodal_inputs,
    load_gemma3_model,
    load_meme_samples,
    load_residual_stats,
    make_joint_reconstruction_hook,
    resolve_target_module,
)
from gemma3_experiments.residual_sae.datasets import build_simple_classify_prompt
from gemma3_experiments.residual_sae.gemma import (
    GemmaSae,
    _YesNoLogitsProcessor,
    get_yes_no_token_ids,
    score_label_candidates,
)
from gemma3_experiments.residual_sae.models import load_residual_sae

logger = logging.getLogger(__name__)


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    if path.name == "best_finite":
        return 10**13, path.name
    if path.name == "last_finite":
        return 10**12 + 1, path.name
    if path.name.startswith("step_"):
        try:
            return int(path.name.split("_", maxsplit=1)[1]), path.name
        except ValueError:
            return -1, path.name
    if path.name == "final":
        return 10**12, path.name
    return -1, path.name


def _checkpoint_is_finite(path: Path) -> bool:
    weights_fpath = path / "sae.safetensors"
    assert weights_fpath.exists(), f"Missing checkpoint weights: {weights_fpath}"
    tensors = load_file(str(weights_fpath))
    return all(torch.isfinite(tensor).all() for tensor in tensors.values())


def _resolve_residual_sae_path(requested_path: str) -> Path:
    resolved_path = Path(requested_path)
    assert resolved_path.exists(), f"Residual SAE checkpoint not found: {resolved_path}"

    if _checkpoint_is_finite(resolved_path):
        return resolved_path

    parent = resolved_path.parent
    candidates = sorted(
        (path for path in parent.iterdir() if path.is_dir() and (path / "sae.safetensors").exists()),
        key=_checkpoint_sort_key,
        reverse=True,
    )
    for candidate in candidates:
        if candidate == resolved_path:
            continue
        if _checkpoint_is_finite(candidate):
            logger.warning(
                "Residual SAE checkpoint %s contains non-finite weights; falling back to %s",
                resolved_path,
                candidate,
            )
            return candidate

    raise RuntimeError(
        f"Residual SAE checkpoint {resolved_path} contains non-finite weights and no finite fallback was found "
        f"in {parent}"
    )


def _is_binary_task(sample) -> bool:
    return len(sample.schema.primary_field.labels) == 2


def _decode_strategy_for_sample(cfg: ResidualInferenceConfig, sample) -> str:
    if not cfg.constrained_decode or cfg.decode_strategy == "free_generate":
        return "free_generate"
    if cfg.decode_strategy == "yes_no":
        return "yes_no"
    if cfg.decode_strategy == "label_score":
        return "label_score"
    if cfg.decode_strategy == "hierarchical_label_score":
        return "hierarchical_label_score"
    if _is_binary_task(sample):
        return "yes_no"
    return "label_score"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(ResidualInferenceConfig)
    samples = load_meme_samples(cfg.dataset)
    if cfg.binary_only:
        before = len(samples)
        samples = [sample for sample in samples if _is_binary_task(sample)]
        logger.info("Binary-only filtering kept %d/%d samples.", len(samples), before)
    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]

    residual_sae_path: Path | None = None
    if cfg.apply_residual_sae:
        residual_sae_path = _resolve_residual_sae_path(cfg.residual_sae_path)
        assert cfg.reconstruction_alpha >= 0.0, "reconstruction_alpha must be non-negative"
    else:
        assert cfg.steering_feature_idx is None, "steering_feature_idx requires apply_residual_sae=true"
        assert not cfg.steering_feature_indices, "steering_feature_indices require apply_residual_sae=true"
        assert cfg.steering_delta == 0.0, "steering_delta requires apply_residual_sae=true"
        assert not cfg.steering_deltas, "steering_deltas require apply_residual_sae=true"
        assert not cfg.steering_values, "steering_values require apply_residual_sae=true"
    logger.info(
        "Starting joint inference: n_samples=%d, apply_residual_sae=%s, residual_sae=%s, requested_residual_sae=%s, "
        "output=%s, model=%s, splits=%s, steering_feature_idx=%s, steering_delta=%s, constrained_decode=%s, "
        "decode_strategy=%s, binary_only=%s, reconstruction_alpha=%s",
        len(samples),
        cfg.apply_residual_sae,
        residual_sae_path,
        cfg.residual_sae_path,
        cfg.output_path,
        cfg.gemma.model_id,
        cfg.dataset.splits,
        cfg.steering_feature_idx,
        cfg.steering_delta,
        cfg.constrained_decode,
        cfg.decode_strategy,
        cfg.binary_only,
        cfg.reconstruction_alpha,
    )

    model, processor = load_gemma3_model(cfg.gemma)
    target_module = None
    hook = None
    if cfg.apply_residual_sae:
        assert residual_sae_path is not None
        target_module = resolve_target_module(model, cfg.gemma)
        base_sae = GemmaSae.load_from_hub(
            cfg.gemma.sae_repo_id,
            cfg.gemma.sae_hookpoint,
            device=cfg.gemma.model_device,
        )
        residual_sae = load_residual_sae(residual_sae_path, device=cfg.gemma.model_device)
        residual_stats = load_residual_stats(residual_sae_path)
        logger.info(
            "Models loaded. base_sae and residual_sae both on model_device=%s. residual_stats=%s. Running inference...",
            cfg.gemma.model_device,
            "enabled" if residual_stats is not None else "disabled",
        )
        hook = make_joint_reconstruction_hook(
            base_sae,
            residual_sae,
            feature_idx=cfg.steering_feature_idx,
            feature_indices=cfg.steering_feature_indices or None,
            delta=cfg.steering_delta,
            feature_deltas=cfg.steering_deltas or None,
            feature_values=cfg.steering_values or None,
            steering_mode=cfg.steering_mode,
            reconstruction_alpha=cfg.reconstruction_alpha,
            residual_stats=residual_stats,
        )
    else:
        logger.info("Models loaded. Running baseline inference without the residual SAE hook.")

    # Pre-resolve yes/no token IDs once so the tokenizer lookup is not repeated per sample.
    if cfg.constrained_decode:
        yes_ids, no_ids = get_yes_no_token_ids(processor)
        yn_constraint = _YesNoLogitsProcessor(yes_ids + no_ids)
        logger.info("Constrained decode enabled. yes_ids=%s no_ids=%s", yes_ids, no_ids)

    output_fpath = Path(cfg.output_path)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    log_every = max(1, len(samples) // 20)
    with output_fpath.open("w") as fd:
        for sample_i, sample in enumerate(samples, start=1):
            image = Image.open(sample.image_path).convert("RGB")
            handle = target_module.register_forward_hook(hook) if hook is not None and target_module is not None else None
            try:
                strategy = _decode_strategy_for_sample(cfg, sample)
                if strategy == "yes_no":
                    # Constrained yes/no classification — immune to JSON collapse under SAE intervention.
                    # The logits processor forces the model to output only a yes/no token, so
                    # prediction_text is always exactly "yes" or "no" regardless of steering strength.
                    prompt_text = build_simple_classify_prompt(sample)
                    inputs = build_multimodal_inputs(processor, image, prompt_text, cfg.gemma.model_device)
                    with torch.no_grad():
                        gen_out = model.generate(
                            **inputs,
                            max_new_tokens=1,
                            do_sample=False,
                            output_scores=True,
                            return_dict_in_generate=True,
                            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                            logits_processor=[yn_constraint],
                        )
                    logits_first = gen_out.scores[0][0].float().cpu()
                    logit_yes = max(float(logits_first[i]) for i in yes_ids)
                    logit_no = max(float(logits_first[i]) for i in no_ids)
                    is_positive = logit_yes > logit_no
                    conf_gap = float(torch.sigmoid(torch.tensor(logit_yes - logit_no)))
                    prediction_text = "yes" if is_positive else "no"
                    # Format prediction label using schema's positive class name.
                    primary_labels = sample.schema.primary_field.labels
                    pred_label = primary_labels[-1] if is_positive and primary_labels else (
                        primary_labels[0] if primary_labels else prediction_text
                    )
                    row = {
                        **sample.to_manifest_dict(),
                        "prompt_text": prompt_text,
                        "prediction_text": prediction_text,
                        "prediction_label": pred_label,
                        "is_positive": is_positive,
                        "conf_gap": round(conf_gap, 4),
                        "constrained_decode": True,
                        "decode_strategy": "yes_no",
                        "apply_residual_sae": cfg.apply_residual_sae,
                        "reconstruction_alpha": cfg.reconstruction_alpha if cfg.apply_residual_sae else 0.0,
                        "residual_sae_path": str(residual_sae_path) if residual_sae_path is not None else None,
                        "steering_feature_idx": cfg.steering_feature_idx,
                        "steering_delta": cfg.steering_delta,
                    }
                elif strategy == "label_score":
                    prompt_text = build_primary_label_prompt(sample)
                    label_texts = list(sample.schema.primary_field.labels)
                    prediction_label, label_scores, prediction_confidence = score_label_candidates(
                        model,
                        processor,
                        image,
                        prompt_text,
                        label_texts,
                        cfg.gemma.model_device,
                    )
                    row = {
                        **sample.to_manifest_dict(),
                        "prompt_text": prompt_text,
                        "prediction_text": prediction_label,
                        "prediction_label": prediction_label,
                        "candidate_labels": label_texts,
                        "label_scores": {key: round(value, 4) for key, value in label_scores.items()},
                        "prediction_confidence": round(prediction_confidence, 4),
                        "constrained_decode": True,
                        "decode_strategy": "label_score",
                        "apply_residual_sae": cfg.apply_residual_sae,
                        "reconstruction_alpha": cfg.reconstruction_alpha if cfg.apply_residual_sae else 0.0,
                        "residual_sae_path": str(residual_sae_path) if residual_sae_path is not None else None,
                        "steering_feature_idx": cfg.steering_feature_idx,
                        "steering_delta": cfg.steering_delta,
                    }
                elif strategy == "hierarchical_label_score":
                    assert sample.dataset_name == "MMHS150K", (
                        "hierarchical_label_score is currently defined only for MMHS150K."
                    )
                    assert sample.task_name == "hate_category", (
                        "hierarchical_label_score requires the MMHS150K hate_category task."
                    )
                    text_ctx = sample.meme_text or "[none]"
                    stage1_prompt = (
                        f"This is a meme.\nMeme text: {text_ctx!r}\n"
                        "Task: decide whether the meme contains hateful content.\n"
                        "Choose exactly one label from: 'NotHate', 'Hate'.\n"
                        "Answer only with the label text."
                    )
                    stage1_label, stage1_scores, stage1_confidence = score_label_candidates(
                        model,
                        processor,
                        image,
                        stage1_prompt,
                        ["NotHate", "Hate"],
                        cfg.gemma.model_device,
                    )
                    stage2_prompt = build_primary_label_prompt(sample)
                    hate_labels = [
                        label for label in sample.schema.primary_field.labels if label != "NotHate"
                    ]
                    stage2_label, stage2_scores, stage2_confidence = score_label_candidates(
                        model,
                        processor,
                        image,
                        stage2_prompt,
                        hate_labels,
                        cfg.gemma.model_device,
                    )
                    prediction_label = "NotHate" if stage1_label == "NotHate" else stage2_label
                    row = {
                        **sample.to_manifest_dict(),
                        "prompt_text": stage2_prompt,
                        "prediction_text": prediction_label,
                        "prediction_label": prediction_label,
                        "candidate_labels": list(sample.schema.primary_field.labels),
                        "stage1_label": stage1_label,
                        "stage1_label_scores": {key: round(value, 4) for key, value in stage1_scores.items()},
                        "stage1_prediction_confidence": round(stage1_confidence, 4),
                        "stage2_label": stage2_label,
                        "stage2_label_scores": {key: round(value, 4) for key, value in stage2_scores.items()},
                        "stage2_prediction_confidence": round(stage2_confidence, 4),
                        "constrained_decode": True,
                        "decode_strategy": "hierarchical_label_score",
                        "apply_residual_sae": cfg.apply_residual_sae,
                        "reconstruction_alpha": cfg.reconstruction_alpha if cfg.apply_residual_sae else 0.0,
                        "residual_sae_path": str(residual_sae_path) if residual_sae_path is not None else None,
                        "steering_feature_idx": cfg.steering_feature_idx,
                        "steering_delta": cfg.steering_delta,
                    }
                else:
                    # Free-generation mode (legacy). Can collapse to malformed JSON under strong SAE interventions.
                    prompt_text = build_classification_prompt(sample)
                    inputs = build_multimodal_inputs(processor, image, prompt_text, cfg.gemma.model_device)
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=cfg.gemma.max_new_tokens,
                            do_sample=False,
                            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                        )
                    input_len = inputs["input_ids"].shape[1]
                    generated_text = processor.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()
                    row = {
                        **sample.to_manifest_dict(),
                        "prompt_text": prompt_text,
                        "prediction_text": generated_text,
                        "constrained_decode": False,
                        "decode_strategy": "free_generate",
                        "apply_residual_sae": cfg.apply_residual_sae,
                        "reconstruction_alpha": cfg.reconstruction_alpha if cfg.apply_residual_sae else 0.0,
                        "residual_sae_path": str(residual_sae_path) if residual_sae_path is not None else None,
                        "steering_feature_idx": cfg.steering_feature_idx,
                        "steering_delta": cfg.steering_delta,
                    }
            finally:
                if handle is not None:
                    handle.remove()

            fd.write(json.dumps(row) + "\n")

            if sample_i == 1 or sample_i == len(samples) or sample_i % log_every == 0:
                elapsed = time.monotonic() - t_start
                rate = sample_i / elapsed if elapsed > 0 else 0.0
                eta = (len(samples) - sample_i) / rate if rate > 0 else float("inf")
                pred_info = f"pred={row.get('prediction_text','?')}"
                if "conf_gap" in row:
                    pred_info += f" conf={row['conf_gap']:.3f}"
                elif "prediction_confidence" in row:
                    pred_info += f" conf={row['prediction_confidence']:.3f}"
                logger.info(
                    "Inference progress: %d/%d (%.1f%%) dataset=%s sample_id=%s strategy=%s %s "
                    "elapsed=%.0fs rate=%.2f samples/s ETA=%.0fs",
                    sample_i,
                    len(samples),
                    100.0 * sample_i / len(samples),
                    sample.dataset_name,
                    sample.sample_id,
                    row.get("decode_strategy", "?"),
                    pred_info,
                    elapsed,
                    rate,
                    eta,
                )

    logger.info(
        "Inference complete: %d samples written to %s in %.0fs",
        len(samples),
        output_fpath,
        time.monotonic() - t_start,
    )


if __name__ == "__main__":
    main()
