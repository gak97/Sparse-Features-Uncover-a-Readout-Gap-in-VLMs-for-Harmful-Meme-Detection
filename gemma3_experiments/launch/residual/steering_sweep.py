import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.evaluate_predictions import _paired_summary, _read_jsonl, _summarize
from gemma3_experiments.launch.residual.joint_infer import _decode_strategy_for_sample, _is_binary_task, _resolve_residual_sae_path
from gemma3_experiments.residual_sae import (
    ResidualSteeringSweepConfig,
    build_multimodal_inputs,
    build_primary_label_prompt,
    load_gemma3_model,
    load_meme_samples,
    load_residual_stats,
    make_joint_reconstruction_hook,
    resolve_target_module,
)
from gemma3_experiments.residual_sae.datasets import build_simple_classify_prompt
from gemma3_experiments.residual_sae.gemma import GemmaSae, _YesNoLogitsProcessor, get_yes_no_token_ids, score_label_candidates
from gemma3_experiments.residual_sae.models import load_residual_sae

logger = logging.getLogger(__name__)


def _load_candidates(cfg: ResidualSteeringSweepConfig) -> list[dict[str, Any]]:
    candidates = _read_jsonl(Path(cfg.candidates_path))
    datasets = set(cfg.dataset.datasets)
    candidate_ids = set(cfg.candidate_ids)
    filtered = [
        row
        for row in candidates
        if row.get("dataset_name") in datasets
        and (not candidate_ids or row.get("candidate_id") in candidate_ids)
        and (cfg.candidate_label is None or row.get("label") == cfg.candidate_label)
        and (cfg.candidate_direction is None or row.get("direction") == cfg.candidate_direction)
    ]
    assert filtered, (
        f"No steering candidates matched config from {cfg.candidates_path}. "
        f"datasets={sorted(datasets)}, candidate_label={cfg.candidate_label!r}, "
        f"candidate_direction={cfg.candidate_direction!r}, n_candidate_ids={len(candidate_ids)}"
    )
    return filtered


def _sample_key(sample) -> tuple[str, str]:
    return (sample.dataset_name, sample.task_name)


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, str]:
    return (candidate["dataset_name"], candidate["task_name"])


def _filter_candidates_for_samples(
    candidates: list[dict[str, Any]],
    samples,
    max_candidates: int | None,
) -> list[dict[str, Any]]:
    sample_counts = Counter(_sample_key(sample) for sample in samples)
    candidate_counts = Counter(_candidate_key(candidate) for candidate in candidates)
    filtered = [candidate for candidate in candidates if _candidate_key(candidate) in sample_counts]
    skipped_counts = candidate_counts - Counter(_candidate_key(candidate) for candidate in filtered)
    if sample_counts:
        logger.info("Loaded eval samples by dataset/task: %s", dict(sample_counts.most_common()))
    if skipped_counts:
        logger.warning(
            "Ignoring %d candidates with no matching eval samples before applying max_candidates: %s",
            sum(skipped_counts.values()),
            dict(skipped_counts.most_common()),
        )
    if max_candidates is not None:
        filtered = filtered[:max_candidates]
    if not filtered:
        raise ValueError(
            "No steering candidates have matching eval samples. "
            f"Available sample keys: {sorted(sample_counts)}. "
            f"Candidate keys: {sorted(candidate_counts)}."
        )
    logger.info("Keeping %d/%d steering candidates after sample matching", len(filtered), len(candidates))
    return filtered


def _samples_for_candidate(samples, candidate: dict[str, Any]):
    return [
        sample
        for sample in samples
        if sample.dataset_name == candidate["dataset_name"]
        and sample.task_name == candidate["task_name"]
    ]


def _candidate_delta(candidate: dict[str, Any], delta_value: float) -> float:
    return float(candidate.get("suggested_sign", 1)) * delta_value


def _decode_one(
    *,
    model,
    processor,
    image: Image.Image,
    sample,
    cfg: ResidualSteeringSweepConfig,
    yn_constraint,
    yes_ids: list[int],
    no_ids: list[int],
) -> dict[str, Any]:
    strategy = _decode_strategy_for_sample(cfg, sample)
    if strategy == "yes_no":
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
        primary_labels = sample.schema.primary_field.labels
        pred_label = primary_labels[-1] if is_positive and primary_labels else (primary_labels[0] if primary_labels else "yes" if is_positive else "no")
        return {
            **sample.to_manifest_dict(),
            "prompt_text": prompt_text,
            "prediction_text": "yes" if is_positive else "no",
            "prediction_label": pred_label,
            "is_positive": is_positive,
            "conf_gap": round(float(torch.sigmoid(torch.tensor(logit_yes - logit_no))), 4),
            "constrained_decode": True,
            "decode_strategy": "yes_no",
        }

    if strategy == "label_score":
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
        return {
            **sample.to_manifest_dict(),
            "prompt_text": prompt_text,
            "prediction_text": prediction_label,
            "prediction_label": prediction_label,
            "candidate_labels": label_texts,
            "label_scores": {key: round(value, 4) for key, value in label_scores.items()},
            "prediction_confidence": round(prediction_confidence, 4),
            "constrained_decode": True,
            "decode_strategy": "label_score",
        }

    raise ValueError("Steering sweep supports constrained yes_no and label_score decoding only")


def run_steering_sweep(cfg: ResidualSteeringSweepConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(cfg)
    samples = load_meme_samples(cfg.dataset)
    if cfg.binary_only:
        samples = [sample for sample in samples if _is_binary_task(sample)]
    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]
    assert samples, "No samples matched steering sweep config"
    candidates = _filter_candidates_for_samples(candidates, samples, cfg.max_candidates)

    residual_sae_path = _resolve_residual_sae_path(cfg.residual_sae_path)
    model, processor = load_gemma3_model(cfg.gemma)
    target_module = resolve_target_module(model, cfg.gemma)
    base_sae = GemmaSae.load_from_hub(cfg.gemma.sae_repo_id, cfg.gemma.sae_hookpoint, device=cfg.gemma.model_device)
    residual_sae = load_residual_sae(residual_sae_path, device=cfg.gemma.model_device)
    residual_stats = load_residual_stats(residual_sae_path)
    yes_ids, no_ids = get_yes_no_token_ids(processor)
    yn_constraint = _YesNoLogitsProcessor(yes_ids + no_ids)
    baseline_rows = _read_jsonl(Path(cfg.baseline_predictions_path)) if cfg.baseline_predictions_path else None

    result_rows = []
    t_start = time.monotonic()
    for candidate_i, candidate in enumerate(candidates, start=1):
        candidate_samples = _samples_for_candidate(samples, candidate)
        if not candidate_samples:
            logger.warning("Skipping candidate with no matching samples: %s", candidate["candidate_id"])
            continue
        for delta_value in cfg.steering_delta_values:
            signed_delta = _candidate_delta(candidate, delta_value)
            if cfg.steering_mode == "clamp":
                feature_values = [candidate["pos_mean"] if signed_delta >= 0 else candidate["neg_mean"]]
                feature_deltas = None
            else:
                feature_values = None
                feature_deltas = [signed_delta]
            hook = make_joint_reconstruction_hook(
                base_sae,
                residual_sae,
                feature_indices=[int(candidate["feature_idx"])],
                feature_deltas=feature_deltas,
                feature_values=feature_values,
                steering_mode=cfg.steering_mode,
                reconstruction_alpha=cfg.reconstruction_alpha,
                residual_stats=residual_stats,
            )
            prediction_rows = []
            for sample in candidate_samples:
                image = Image.open(sample.image_path).convert("RGB")
                handle = target_module.register_forward_hook(hook)
                try:
                    row = _decode_one(
                        model=model,
                        processor=processor,
                        image=image,
                        sample=sample,
                        cfg=cfg,
                        yn_constraint=yn_constraint,
                        yes_ids=yes_ids,
                        no_ids=no_ids,
                    )
                finally:
                    handle.remove()
                row.update(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "feature_idx": int(candidate["feature_idx"]),
                        "candidate_label": candidate["label"],
                        "candidate_direction": candidate["direction"],
                        "steering_delta": signed_delta,
                        "reconstruction_alpha": cfg.reconstruction_alpha,
                        "residual_sae_path": str(residual_sae_path),
                    }
                )
                prediction_rows.append(row)
            summary = _summarize(prediction_rows)
            if baseline_rows is not None:
                summary["paired_baseline"] = _paired_summary(prediction_rows, baseline_rows)
            result = {
                "candidate": candidate,
                "delta_value": delta_value,
                "signed_delta": signed_delta,
                "n_predictions": len(prediction_rows),
                "metrics": summary,
            }
            result_rows.append(result)
            with (output_dir / "steering_sweep_results.jsonl").open("a") as fd:
                fd.write(json.dumps(result) + "\n")
            logger.info(
                "candidate %d/%d delta=%s id=%s accuracy=%.4f elapsed=%.0fs",
                candidate_i,
                len(candidates),
                signed_delta,
                candidate["candidate_id"],
                summary["overall"]["accuracy"],
                time.monotonic() - t_start,
            )

    ranked = sorted(
        result_rows,
        key=lambda row: row["metrics"].get("paired_baseline", {}).get("delta_accuracy", row["metrics"]["overall"]["accuracy"]),
        reverse=True,
    )
    with (output_dir / "steering_sweep_summary.json").open("w") as fd:
        json.dump(
            {
                "config": cfg.to_dict(),
                "n_candidates": len(candidates),
                "n_results": len(result_rows),
                "top_results": ranked[: min(50, len(ranked))],
            },
            fd,
            indent=2,
        )
    return {"output_dir": str(output_dir), "n_results": len(result_rows)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ResidualSteeringSweepConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run_steering_sweep(cfg)


if __name__ == "__main__":
    main()
