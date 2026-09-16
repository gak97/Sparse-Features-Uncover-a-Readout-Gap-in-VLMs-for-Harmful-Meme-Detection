

import gc
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.probe_validate import (
    _load_fhm_confounder_metadata,
    _normalise_fhm_id,
    _normalise_group_value,
    _per_class_metrics,
)
from gemma3_experiments.meme_safety_probe import MemeSafetyProbe, MemeSafetyProbeConfig
from gemma3_experiments.residual_sae import (
    GemmaResidualConfig,
    MemeDatasetConfig,
    MemeSample,
    TaskSchema,
    classify_constrained,
    load_gemma3_model,
    load_meme_samples,
    resolve_visible_device,
    score_label_candidates,
)
from gemma3_experiments.residual_sae.datasets import (
    build_classification_prompt,
    build_primary_label_prompt,
    build_simple_classify_prompt,
)

logger = logging.getLogger(__name__)

_STANDARD_CONDITIONS = (
    "original",
    "blank",
    "shuffled",
    "no_ocr",
    "blank_ocr_only",
    "neutral_description",
)


@dataclass
class VisualCreditAuditConfig:
    dataset: MemeDatasetConfig = field(default_factory=MemeDatasetConfig)
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    probe: MemeSafetyProbeConfig = field(default_factory=MemeSafetyProbeConfig)

    task_name: str = ""
    max_samples: int | None = None
    shuffle_seed: int = 42
    positive_labels: list[str] = field(default_factory=list)
    collapse_harmful: bool = False
    fhm_confounder_metadata_path: str = ""

    run_native: bool = True
    lora_adapter_path: str = ""
    lora_protocol: str = ""
    run_probe: bool = False
    output_dir: str = "./visual_credit_audit"


def _slug(text: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def _normalise_probs(scores: dict[str, float]) -> dict[str, float]:
    values = torch.tensor(list(scores.values()), dtype=torch.float32)
    probabilities = torch.softmax(values, dim=0).tolist()
    return {label: float(probability) for label, probability in zip(scores, probabilities)}


def _prompt(sample: MemeSample, condition: str, system_name: str) -> str:
    if condition in {"original", "blank", "shuffled", "no_ocr"}:
        if system_name == "probe":
            return build_classification_prompt(sample)
        return build_simple_classify_prompt(sample) if len(sample.schema.primary_field.labels) == 2 else build_primary_label_prompt(sample)

    text = sample.meme_text or "[no OCR text available]"
    if condition == "blank_ocr_only":
        answer = "yes or no" if len(sample.schema.primary_field.labels) == 2 else "one label"
        return f'OCR text: "{text}"\nAnswer only with {answer}.'
    assert condition == "neutral_description", f"Unsupported condition: {condition}"
    return "Describe the image and any visible text neutrally. Do not assess harm, offensiveness, or intent.\nResponse:"


def _make_blank_image(sample: MemeSample, blank_dir: Path) -> str:
    destination = blank_dir / f"{_slug(sample.dataset_name)}__{_slug(sample.sample_id)}.png"
    if destination.exists():
        return str(destination)
    with Image.open(sample.image_path) as image:
        blank = Image.new("RGB", image.size, color=(128, 128, 128))
    blank.save(destination)
    return str(destination)


def _deranged_image_paths(samples: list[MemeSample], seed: int) -> dict[str, str]:
    assert len(samples) >= 2, "At least two samples are required for same-split image shuffling."
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    for offset in range(len(shuffled)):
        candidate = shuffled[offset:] + shuffled[:offset]
        if all(source.sample_id != target.sample_id for source, target in zip(ordered, candidate)):
            return {sample.sample_id: replacement.image_path for sample, replacement in zip(ordered, candidate)}
    raise AssertionError("Could not construct a fixed-point-free same-split image permutation.")


def _condition_inputs(
    sample: MemeSample,
    condition: str,
    system_name: str,
    blank_dir: Path,
    shuffled_images: dict[str, str],
) -> tuple[str, str, str | None, str | None]:
    meme_text = "" if condition == "no_ocr" else sample.meme_text
    prompt_sample = replace(sample, meme_text=meme_text)
    prompt_text = _prompt(prompt_sample, condition, system_name)
    if condition in {"blank", "blank_ocr_only"}:
        return _make_blank_image(sample, blank_dir), meme_text, prompt_text, None
    if condition == "shuffled":
        return shuffled_images[sample.sample_id], meme_text, prompt_text, shuffled_images[sample.sample_id]
    return sample.image_path, meme_text, prompt_text, None


def _model_prediction(
    model: Any,
    processor: Any,
    device: str,
    schema: TaskSchema,
    image_path: str,
    prompt_text: str,
) -> tuple[str, dict[str, float], float]:
    labels = schema.primary_field.labels
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if len(labels) == 2:
            is_positive, _, confidence = classify_constrained(model, processor, image, prompt_text, device)
            probabilities = {labels[-1]: confidence, labels[0]: 1.0 - confidence}
            return (labels[-1] if is_positive else labels[0]), probabilities, confidence
        prediction, log_scores, confidence = score_label_candidates(model, processor, image, prompt_text, labels, device)
    return prediction, _normalise_probs(log_scores), confidence


def _probe_prediction(
    probe: MemeSafetyProbe,
    schema: TaskSchema,
    image_path: str,
    meme_text: str,
    prompt_text: str,
) -> tuple[str, dict[str, float], float | None]:
    result = probe.predict(image_path, meme_text, schema=schema, prompt_override=prompt_text)
    assert result.label_scores is not None, "The visual-credit audit requires a probabilistic probe classifier."
    return result.label, result.label_scores, result.confidence


def _evaluate_system(
    *,
    system_name: str,
    samples: list[MemeSample],
    conditions: tuple[str, ...],
    predictor: Any,
    blank_dir: Path,
    shuffled_images: dict[str, str],
    collapse_harmful: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(samples) * len(conditions)
    for sample_i, sample in enumerate(samples, start=1):
        gold = sample.gold_fields.get(sample.schema.primary_field.name)
        assert isinstance(gold, str), f"Missing primary gold label for {sample.sample_id}"
        for condition in conditions:
            image_path, meme_text, prompt_text, shuffled_image_path = _condition_inputs(
                sample, condition, system_name, blank_dir, shuffled_images,
            )
            prediction, probabilities, confidence = predictor(sample.schema, image_path, meme_text, prompt_text)
            assert set(probabilities) == set(sample.schema.primary_field.labels), (
                f"{system_name} scores do not match schema labels for {sample.sample_id}: {probabilities}"
            )
            if collapse_harmful:
                assert set(probabilities) == {"not harmful", "somewhat harmful", "very harmful"}, (
                    "collapse_harmful only supports the HarMeme primary harmfulness labels."
                )
                gold = "not harmful" if gold == "not harmful" else "harmful"
                prediction = "not harmful" if prediction == "not harmful" else "harmful"
                probabilities = {
                    "not harmful": probabilities["not harmful"],
                    "harmful": probabilities["somewhat harmful"] + probabilities["very harmful"],
                }
            rows.append({
                "system": system_name,
                "condition": condition,
                "sample_id": sample.sample_id,
                "dataset_name": sample.dataset_name,
                "task_name": sample.task_name,
                "gold": gold,
                "pred": prediction,
                "correct": prediction == gold,
                "gold_probability": round(float(probabilities[gold]), 6),
                "label_probabilities": {label: round(float(score), 6) for label, score in probabilities.items()},
                "confidence": None if confidence is None else round(float(confidence), 6),
                "prompt_text": prompt_text,
                "image_path": image_path,
                "shuffled_image_path": shuffled_image_path,
            })
        if sample_i % max(1, len(samples) // 20) == 0:
            logger.info("%s: processed %d/%d samples (%d/%d condition rows)", system_name, sample_i, len(samples), len(rows), total)
    return rows


def _condition_metrics(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    subset = [row for row in rows if row["condition"] == condition]
    return _per_class_metrics([row["gold"] for row in subset], [row["pred"] for row in subset])


def _fhm_pair_ordering(rows: list[dict[str, Any]], metadata_path: Path) -> dict[str, Any]:
    metadata = _load_fhm_confounder_metadata(metadata_path)
    original_rows = [row for row in rows if row["condition"] == "original"]
    ids = {row["sample_id"] for row in original_rows}
    result: dict[str, Any] = {}
    for source, column in {"pseudo_img": "pseudo_img_idx", "pseudo_text": "pseudo_text_idx"}.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in original_rows:
            value = _normalise_group_value(metadata.get(_normalise_fhm_id(row["sample_id"]), {}).get(column))
            if value is not None:
                groups[value].append(row)
        by_condition: dict[str, Any] = {}
        for condition in _STANDARD_CONDITIONS:
            score_by_id = {
                row["sample_id"]: row["label_probabilities"].get("hateful", 0.0)
                for row in rows if row["condition"] == condition and row["sample_id"] in ids
            }
            correct_orders = 0
            n_pairs = 0
            n_groups = 0
            for group_rows in groups.values():
                positive = [row["sample_id"] for row in group_rows if row["gold"].lower() == "hateful"]
                negative = [row["sample_id"] for row in group_rows if row["gold"].lower() == "not hateful"]
                if not positive or not negative:
                    continue
                n_groups += 1
                for positive_id in positive:
                    for negative_id in negative:
                        n_pairs += 1
                        correct_orders += score_by_id[positive_id] > score_by_id[negative_id]
            by_condition[condition] = {
                "n_cross_label_groups": n_groups,
                "n_pairs": n_pairs,
                "positive_pair_ordering_rate": round(correct_orders / n_pairs, 4) if n_pairs else None,
            }
        result[source] = by_condition
    return result


def _report_system(
    rows: list[dict[str, Any]], positive_labels: set[str], fhm_metadata_path: str,
) -> dict[str, Any]:
    by_condition = {condition: _condition_metrics(rows, condition) for condition in _STANDARD_CONDITIONS}
    original = {row["sample_id"]: row for row in rows if row["condition"] == "original"}
    controls: dict[str, Any] = {}
    for condition in _STANDARD_CONDITIONS[1:]:
        control = {row["sample_id"]: row for row in rows if row["condition"] == condition}
        paired = [(original[sample_id], control[sample_id]) for sample_id in original]
        gold_margin_drop = sum(left["gold_probability"] - right["gold_probability"] for left, right in paired) / len(paired)
        original_negative = [left for left, _ in paired if left["gold"] not in positive_labels]
        original_fp_rate = (
            sum(left["pred"] in positive_labels for left in original_negative) / len(original_negative)
            if original_negative else None
        )
        control_fp_rate = (
            sum(right["pred"] in positive_labels for left, right in paired if left["gold"] not in positive_labels)
            / len(original_negative) if original_negative else None
        )
        controls[condition] = {
            "original_minus_control_gold_label_margin": round(gold_margin_drop, 6),
            "original_false_positive_rate": round(original_fp_rate, 6) if original_fp_rate is not None else None,
            "control_false_positive_rate": round(control_fp_rate, 6) if control_fp_rate is not None else None,
            "false_positive_rate_change": round(control_fp_rate - original_fp_rate, 6)
            if control_fp_rate is not None and original_fp_rate is not None else None,
        }

    blank = {row["sample_id"]: row for row in rows if row["condition"] == "blank"}
    paired_blank = [(original[sample_id], blank[sample_id]) for sample_id in original]
    original_correct = [pair for pair in paired_blank if pair[0]["correct"]]
    visual_credit = {
        "correct_but_image_uncredited_rate": round(
            sum(control["correct"] for _, control in original_correct) / len(original_correct), 6,
        ) if original_correct else None,
        "image_credited_correctness": round(
            sum(left["correct"] and not right["correct"] for left, right in paired_blank) / len(paired_blank), 6,
        ),
        "definition": "blank is the same-size neutral-gray image with the original OCR/task prompt",
    }
    report: dict[str, Any] = {
        "ordinary_macro_f1": {condition: metrics["macro_f1"] for condition, metrics in by_condition.items()},
        "condition_metrics": by_condition,
        "original_vs_control": controls,
        "visual_credit": visual_credit,
    }
    if fhm_metadata_path:
        report["fhm_pair_response"] = _fhm_pair_ordering(rows, Path(fhm_metadata_path))
    return report


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(cfg: VisualCreditAuditConfig) -> None:
    assert cfg.run_native or cfg.lora_adapter_path or cfg.run_probe, "Enable at least one system."
    out_dir = Path(cfg.output_dir)
    blank_dir = out_dir / "blank_images"
    blank_dir.mkdir(parents=True, exist_ok=True)

    samples = load_meme_samples(cfg.dataset)
    if cfg.task_name:
        samples = [sample for sample in samples if sample.task_name == cfg.task_name]
    assert samples, "No labeled samples found for the requested dataset/split/task."
    assert len({sample.dataset_name for sample in samples}) == 1 and len({sample.task_name for sample in samples}) == 1, (
        "Run one dataset task per audit so shuffled images remain same-split and same-task."
    )
    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]
    assert len(samples) >= 2, "Need at least two samples for the shuffled-image control."
    assert all(isinstance(sample.gold_fields.get(sample.schema.primary_field.name), str) for sample in samples)

    dataset_name = samples[0].dataset_name
    task_name = samples[0].task_name
    labels = ("not harmful", "harmful") if cfg.collapse_harmful else samples[0].schema.primary_field.labels
    if cfg.collapse_harmful:
        assert dataset_name == "HarMeme", "collapse_harmful is only valid for HarMeme."
    positive_labels = set(cfg.positive_labels or [labels[-1]])
    assert positive_labels <= set(labels), f"positive_labels={sorted(positive_labels)} are not all schema labels={labels}"
    shuffled_images = _deranged_image_paths(samples, cfg.shuffle_seed)
    logger.info("Visual credit audit: %s/%s, n=%d, labels=%s", dataset_name, task_name, len(samples), labels)

    rows_by_system: dict[str, list[dict[str, Any]]] = {}
    device = resolve_visible_device(cfg.gemma.model_device)
    if cfg.run_native:
        model, processor = load_gemma3_model(cfg.gemma)
        rows_by_system["native"] = _evaluate_system(
            system_name="native", samples=samples, conditions=_STANDARD_CONDITIONS,
            predictor=lambda schema, image_path, _text, prompt_text: _model_prediction(
                model, processor, device, schema, image_path, prompt_text,
            ), blank_dir=blank_dir, shuffled_images=shuffled_images, collapse_harmful=cfg.collapse_harmful,
        )
        _release_model(model)

    if cfg.lora_adapter_path:
        model, processor = load_gemma3_model(cfg.gemma)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.lora_adapter_path).merge_and_unload()
        model.eval()
        rows_by_system["lora"] = _evaluate_system(
            system_name="lora", samples=samples, conditions=_STANDARD_CONDITIONS,
            predictor=lambda schema, image_path, _text, prompt_text: _model_prediction(
                model, processor, device, schema, image_path, prompt_text,
            ), blank_dir=blank_dir, shuffled_images=shuffled_images, collapse_harmful=cfg.collapse_harmful,
        )
        _release_model(model)

    if cfg.run_probe:
        probe = MemeSafetyProbe(cfg.probe)
        rows_by_system["probe"] = _evaluate_system(
            system_name="probe", samples=samples, conditions=_STANDARD_CONDITIONS,
            predictor=lambda schema, image_path, meme_text, prompt_text: _probe_prediction(
                probe, schema, image_path, meme_text, prompt_text,
            ), blank_dir=blank_dir, shuffled_images=shuffled_images, collapse_harmful=cfg.collapse_harmful,
        )
        _release_model(probe.model)

    report = {
        "dataset_name": dataset_name,
        "task_name": task_name,
        "split": sorted({sample.split for sample in samples}),
        "n_samples": len(samples),
        "shuffle_seed": cfg.shuffle_seed,
        "collapse_harmful": cfg.collapse_harmful,
        "lora_protocol": cfg.lora_protocol if cfg.lora_adapter_path else None,
        "positive_labels_for_false_positive_metrics": sorted(positive_labels),
        "conditions": list(_STANDARD_CONDITIONS),
        "condition_notes": {
            "blank": "same-size neutral-gray image with the original OCR and task prompt",
            "shuffled": "fixed-point-free deterministic image permutation within this audited split/task",
            "no_ocr": "original image and standard task prompt with meme text removed",
            "blank_ocr_only": "blank image and OCR-only prompt; prompt-prior diagnostic, not a valid classifier",
            "neutral_description": "original image and neutral-description prompt; prompt-prior diagnostic, not a valid classifier",
        },
        "systems": {},
    }
    for system_name, rows in rows_by_system.items():
        with (out_dir / f"predictions_{system_name}.jsonl").open("w") as fd:
            for row in rows:
                fd.write(json.dumps(row) + "\n")
        report["systems"][system_name] = _report_system(rows, positive_labels, cfg.fhm_confounder_metadata_path)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Wrote visual-credit report to %s", out_dir / "report.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(VisualCreditAuditConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()