"""Test whether MultiBully probe features causally affect Qwen's output route."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from simple_parsing import Serializable, parse

from ..config import QwenResidualConfig
from ..qwen import QwenSae, classify_constrained, load_qwen3_model, resolve_target_module
from ..task_labels import fit_scaled_logreg
from .analyze_exist_multilingual import _extract_features, _read_jsonl


@dataclass
class MultiBullyCausalConfig(Serializable):
    cache_dir: str = "./outputs/multibully/dense_raw"
    predictions_path: str = "./outputs/multibully/baseline_raw_ocr/baseline_eval.jsonl"
    output_dir: str = "./outputs/multibully/causal_raw_ocr"
    device: str = "cuda:0"
    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    hook_module_path: str = "model.language_model.layers.20"
    C: float = 1.0
    max_iter: int = 3000
    top_k_features: int = 20
    n_examples: int = 15


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1 - 1e-6)
    return math.log(probability / (1 - probability))


def _knockout_hook(sae: QwenSae, feature_indices: np.ndarray):
    indices = None

    def hook(_module, _inputs, output):
        nonlocal indices
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            state = hidden.detach().to(torch.float32)
            if indices is None or indices.device != state.device:
                indices = torch.as_tensor(feature_indices, dtype=torch.long, device=state.device)
            acts, _ = sae.encode(state)
            original = sae.decode(acts)
            ablated = acts.clone()
            ablated[..., indices] = 0.0
            updated = state + sae.decode(ablated) - original
        if isinstance(output, tuple):
            return (updated.to(hidden.dtype), *output[1:])
        return updated.to(hidden.dtype)

    return hook


def _score_with_hook(model, processor, target_module, row: dict, device: str, hook) -> float:
    image = Image.open(row["image_path"]).convert("RGB")
    handle = target_module.register_forward_hook(hook)
    try:
        _prediction, _text, confidence = classify_constrained(model, processor, image, row["prompt_text"], device)
        return _logit(confidence)
    finally:
        handle.remove()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(MultiBullyCausalConfig)
    cache_dir, output_dir = Path(cfg.cache_dir), Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [row for row in _read_jsonl(cache_dir / "manifest.jsonl") if row.get("task_name") == "cyberbullying_detection"]
    sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    metadata, features = _extract_features(sae, cache_dir, manifest, cfg.device)
    labels = np.asarray([int(row["gold_fields"]["label"] == "bully") for row in metadata], dtype=np.int32)
    train = np.asarray([index for index, row in enumerate(metadata) if row["split"] == "probe_train"], dtype=np.int64)
    evaluation = np.asarray([index for index, row in enumerate(metadata) if row["split"] == "eval"], dtype=np.int64)
    classifier, scaler = fit_scaled_logreg(features["all"][train], labels[train], C=cfg.C, max_iter=cfg.max_iter)
    probe_scores = classifier.decision_function(scaler.transform(features["all"][evaluation]))
    probe_prediction = (probe_scores > 0).astype(np.int32)
    coefficient = classifier.coef_[0].copy()
    if list(classifier.classes_)[1] != 1:
        coefficient *= -1
    feature_indices = np.argsort(-coefficient)[:cfg.top_k_features]

    baseline = {str(row["sample_id"]): row for row in _read_jsonl(Path(cfg.predictions_path))}
    candidates: list[tuple[int, float]] = []
    for local_index, global_index in enumerate(evaluation):
        row = metadata[global_index]
        prediction = baseline.get(str(row["sample_id"]))
        if prediction is None:
            continue
        if labels[global_index] == 1 and probe_prediction[local_index] == 1 and prediction["prediction"] == "non-bully":
            candidates.append((global_index, float(probe_scores[local_index])))
    candidates.sort(key=lambda item: item[1], reverse=True)
    if not candidates:
        raise ValueError("No gold-bully, probe-correct, baseline-wrong examples available for causal knockout")

    qwen_cfg = QwenResidualConfig(model_id=cfg.model_id, model_device=cfg.device, hook_module_path=cfg.hook_module_path)
    model, processor = load_qwen3_model(qwen_cfg)
    target_module = resolve_target_module(model, qwen_cfg)
    hook = _knockout_hook(sae, feature_indices)
    rows: list[dict] = []
    for global_index, probe_score in candidates[:cfg.n_examples]:
        row = metadata[global_index]
        ablated_vector = features["all"][global_index].copy()
        ablated_vector[feature_indices] = 0.0
        ablated_probe = float(classifier.decision_function(scaler.transform(ablated_vector[None]))[0])
        baseline_logit = _logit(baseline[str(row["sample_id"])]["conf_gap"])
        ablated_logit = _score_with_hook(model, processor, target_module, row, cfg.device, hook)
        rows.append({
            "sample_id": row["sample_id"],
            "gold": "bully",
            "baseline_probe_score": probe_score,
            "ablated_probe_score": ablated_probe,
            "delta_probe": ablated_probe - probe_score,
            "baseline_logit": baseline_logit,
            "ablated_logit": ablated_logit,
            "delta_logit": ablated_logit - baseline_logit,
            "top_feature_indices": [int(index) for index in feature_indices],
        })
        logging.info("%s: delta_probe=%.3f delta_logit=%.3f", row["sample_id"], ablated_probe - probe_score, ablated_logit - baseline_logit)

    with (output_dir / "causal_mediation_results.jsonl").open("w") as fd:
        for row in rows:
            fd.write(json.dumps(row) + "\n")
    summary = {
        "selection": "gold_bully_probe_correct_baseline_wrong",
        "candidate_count": len(candidates),
        "n_examples": len(rows),
        "top_feature_indices": [int(index) for index in feature_indices],
        "mean_abs_delta_probe": round(float(np.mean([abs(row["delta_probe"]) for row in rows])), 4),
        "mean_abs_delta_logit": round(float(np.mean([abs(row["delta_logit"]) for row in rows])), 4),
    }
    (output_dir / "causal_mediation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
