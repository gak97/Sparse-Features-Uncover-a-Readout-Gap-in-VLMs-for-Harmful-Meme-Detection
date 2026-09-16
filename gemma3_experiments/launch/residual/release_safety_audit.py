"""Paired release-safety and capability audit for Gemma LoRA adapters."""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse
from sklearn.metrics import average_precision_score, f1_score


from gemma3_experiments.residual_sae import (
    GemmaResidualConfig,
    MemeSample,
    build_multimodal_inputs,
    classify_constrained,
    load_gemma3_model,
    resolve_visible_device,
)
from gemma3_experiments.residual_sae.datasets import HATEFUL_MEMES_SCHEMA, build_simple_classify_prompt

logger = logging.getLogger(__name__)


@dataclass
class ReleaseSafetyAuditConfig:
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    controls_path: str = ""
    capability_manifest_path: str = ""
    adapter_paths: list[str] = field(default_factory=list)  # name=/absolute/adapter/path
    route_task_to_variant: list[str] = field(default_factory=list)  # dataset::task=name
    max_new_tokens: int = 32
    ece_bins: int = 10
    output_dir: str = "./release_safety_audit"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fd:
        return [json.loads(line) for line in fd if line.strip()]


def _parse_assignments(entries: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        name, value = entry.split("=", maxsplit=1)
        assert name and value, f"Expected NAME=VALUE, got {entry!r}"
        result[name] = value
    return result


def _ece(gold: list[int], prob: list[float], n_bins: int) -> float:
    values = np.asarray(prob, dtype=float)
    labels = np.asarray(gold, dtype=float)
    result = 0.0
    for lower, upper in zip(np.linspace(0, 1, n_bins, endpoint=False), np.linspace(1 / n_bins, 1, n_bins)):
        mask = (values >= lower) & ((values < upper) if upper < 1 else (values <= upper))
        if mask.any():
            result += float(mask.mean() * abs(values[mask].mean() - labels[mask].mean()))
    return result


def _norm(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _caption_score(pred: str, refs: list[str]) -> float:
    pred_tokens = _norm(pred)
    if not pred_tokens:
        return 0.0
    scores = []
    for ref in refs:
        ref_tokens = _norm(ref)
        if not ref_tokens:
            continue
        overlap = len(pred_tokens & ref_tokens)
        scores.append(2 * overlap / (len(pred_tokens) + len(ref_tokens)))
    return max(scores, default=0.0)


def _vqa_score(pred: str, refs: list[str]) -> float:
    normalized = " ".join(_norm(pred))
    return float(any(" ".join(_norm(ref)) in normalized for ref in refs))


def _load_variant(cfg: ReleaseSafetyAuditConfig, adapter_path: str, device: str):
    model, processor = load_gemma3_model(cfg.gemma)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.eval()
    return model, processor


def _safety_rows(model: Any, processor: Any, device: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        sample = MemeSample(
            dataset_name=row["dataset_name"], task_name="release_safety", sample_id=row["sample_id"], split=row["split"],
            image_path=row["image_path"], meme_text=row["meme_text"], schema=HATEFUL_MEMES_SCHEMA, gold_fields={}, raw_fields={},
        )
        is_harmful, output, positive_prob = classify_constrained(
            model, processor, Image.open(sample.image_path).convert("RGB"), build_simple_classify_prompt(sample), device,
        )
        result.append({**row, "pred_harmful": is_harmful, "positive_probability": positive_prob, "output": output, "format_valid": output in {"yes", "no"}})
    return result


def _safety_metrics(rows: list[dict[str, Any]], ece_bins: int) -> dict[str, Any]:
    gold = [int(row["is_harmful"]) for row in rows]
    pred = [int(row["pred_harmful"]) for row in rows]
    prob = [float(row["positive_probability"]) for row in rows]
    by_control: dict[str, dict[str, Any]] = {}
    for control in sorted({row["control"] for row in rows}):
        subset = [row for row in rows if row["control"] == control]
        false_positive = sum(row["pred_harmful"] for row in subset if not row["is_harmful"])
        false_negative = sum(not row["pred_harmful"] for row in subset if row["is_harmful"])
        n_negative = sum(not row["is_harmful"] for row in subset)
        n_positive = sum(row["is_harmful"] for row in subset)
        by_control[control] = {"n": len(subset), "false_positive_rate": false_positive / n_negative if n_negative else None, "false_negative_rate": false_negative / n_positive if n_positive else None}
    return {
        "n": len(rows), "macro_f1": float(f1_score(gold, pred, average="macro")), "pr_auc": float(average_precision_score(gold, prob)),
        "expected_calibration_error": _ece(gold, prob, ece_bins), "output_format_validity": float(np.mean([row["format_valid"] for row in rows])),
        "false_positive_rate_identity_controls": by_control.get("benign_identity_group", {}).get("false_positive_rate"),
        "false_positive_rate_ordinary_images": by_control.get("ordinary_non_meme_image", {}).get("false_positive_rate"),
        "false_negative_rate_direct_threat_slur": by_control.get("direct_threat_or_slur", {}).get("false_negative_rate"), "by_control": by_control,
    }


def _capability_rows(model: Any, processor: Any, device: str, rows: list[dict[str, Any]], max_new_tokens: int) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        inputs = build_multimodal_inputs(processor, Image.open(row["image_path"]).convert("RGB"), row["prompt"], device)
        output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=processor.tokenizer.eos_token_id)
        text = processor.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        score = _caption_score(text, row["reference_answers"]) if row["kind"] == "caption" else _vqa_score(text, row["reference_answers"])
        result.append({**row, "output": text, "score": score, "format_valid": bool(text)})
    return result


def _capability_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        kind: {"n": len(subset), "score": float(np.mean([row["score"] for row in subset])), "output_format_validity": float(np.mean([row["format_valid"] for row in subset]))}
        for kind in sorted({row["kind"] for row in rows})
        if (subset := [row for row in rows if row["kind"] == kind])
    }


def run(cfg: ReleaseSafetyAuditConfig) -> None:
    controls = _read_jsonl(Path(cfg.controls_path))
    capability = _read_jsonl(Path(cfg.capability_manifest_path))
    assert controls and capability, "Both controls_path and capability_manifest_path must contain rows."
    controls.extend({
        "control": "ordinary_non_meme_image", "sample_id": row["sample_id"], "dataset_name": "COCO", "task_name": "captioning_safety",
        "split": "frozen", "image_path": row["image_path"], "meme_text": "", "gold_label": "not harmful", "is_harmful": False,
        "matched_terms": [],
    } for row in capability if row["kind"] == "caption")
    adapter_paths = _parse_assignments(cfg.adapter_paths)
    variants = {"base": "", **adapter_paths}
    routes = _parse_assignments(cfg.route_task_to_variant)
    unknown_routes = {task: variant for task, variant in routes.items() if variant not in variants}
    assert not unknown_routes, f"Routing refers to unknown variants: {unknown_routes}"
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_visible_device(cfg.gemma.model_device)
    per_task_routing = {}
    for task in sorted({f"{row['dataset_name']}::{row['task_name']}" for row in controls}):
        variant = routes.get(task)
        per_task_routing[task] = {
            "configured_variant": variant,
            "routing_configuration_error": variant is not None and variant not in variants,
            "runtime_routing_error_rate": None,
        }
    report: dict[str, Any] = {
        "variants": {},
        "routing": {
            "rules": routes,
            "unknown_route_errors": unknown_routes,
            "per_task": per_task_routing,
            "note": "No runtime router is implemented in this repository; runtime routing error rates remain unreported rather than being imputed as zero.",
        },
    }
    base_safety: dict[str, Any] | None = None
    base_capability: dict[str, Any] | None = None
    for name, path in variants.items():
        model, processor = _load_variant(cfg, path, device)
        safety_rows = _safety_rows(model, processor, device, controls)
        capability_rows = _capability_rows(model, processor, device, capability, cfg.max_new_tokens)
        safety_metrics = _safety_metrics(safety_rows, cfg.ece_bins)
        capability_metrics = _capability_metrics(capability_rows)
        if name == "base":
            base_safety, base_capability = safety_metrics, capability_metrics
        else:
            safety_metrics["delta_macro_f1_vs_base"] = safety_metrics["macro_f1"] - base_safety["macro_f1"]
            capability_metrics = {kind: {**metrics, "delta_score_vs_base": metrics["score"] - base_capability[kind]["score"]} for kind, metrics in capability_metrics.items()}
        with (out_dir / f"predictions_{name}.jsonl").open("w") as fd:
            for row in safety_rows + capability_rows:
                fd.write(json.dumps(row) + "\n")
        report["variants"][name] = {"adapter_path": path or None, "safety": safety_metrics, "capability": capability_metrics}
        del model
    routed_controls = [f"{row['dataset_name']}::{row['task_name']}" for row in controls]
    report["routing"]["n_rows_with_route_rule"] = sum(task in routes for task in routed_controls)
    report["routing"]["n_rows_without_route_rule"] = sum(task not in routes for task in routed_controls)
    (out_dir / "release_safety_audit.json").write_text(json.dumps(report, indent=2))
    logger.info("Wrote paired release audit to %s", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse(ReleaseSafetyAuditConfig, argument_generation_mode=ArgumentGenerationMode.BOTH))


if __name__ == "__main__":
    main()