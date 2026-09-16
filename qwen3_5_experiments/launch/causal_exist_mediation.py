

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
from .analyze_exist_multilingual import _extract_features, _indices, _label, _read_jsonl

logger = logging.getLogger(__name__)


@dataclass
class ExistCausalConfig(Serializable):
    cache_dir: str = "./outputs/exist_multilingual/dense_raw"
    output_dir: str = "./outputs/exist_multilingual/causal_raw_ocr"
    device: str = "cuda:0"
    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    hook_module_path: str = "model.language_model.layers.20"
    C: float = 1.0
    max_iter: int = 3000
    top_k_features: int = 20
    examples_per_language: int = 12


def _logit_from_probability(probability: float) -> float:
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


def _classify(model, processor, image_path: str, prompt: str, device: str, target_module=None, hook=None) -> float:
    image = Image.open(image_path).convert("RGB")
    handle = target_module.register_forward_hook(hook) if hook is not None else None
    try:
        _is_sexist, _text, confidence = classify_constrained(model, processor, image, prompt, device)
        return _logit_from_probability(confidence)
    finally:
        if handle is not None:
            handle.remove()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ExistCausalConfig)
    cache_dir, output_dir = Path(cfg.cache_dir), Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [row for row in _read_jsonl(cache_dir / "manifest.jsonl") if row.get("task_name") == "sexism_detection"]
    sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    metadata, features = _extract_features(sae, cache_dir, manifest, cfg.device)
    X, labels = features["all"], np.asarray([_label(row) for row in metadata], dtype=np.int32)
    train = np.concatenate([_indices(metadata, "probe_train", "en"), _indices(metadata, "probe_train", "es")])
    if not len(train):
        raise ValueError("No bilingual probe_train rows found")
    clf, scaler = fit_scaled_logreg(X[train], labels[train], C=cfg.C, max_iter=cfg.max_iter)
    coefficient = clf.coef_[0].copy()
    if list(clf.classes_)[1] != 1:
        coefficient *= -1
    feature_indices = np.argsort(-coefficient)[:cfg.top_k_features]

    qwen_cfg = QwenResidualConfig(model_id=cfg.model_id, model_device=cfg.device, hook_module_path=cfg.hook_module_path)
    model, processor = load_qwen3_model(qwen_cfg)
    target_module = resolve_target_module(model, qwen_cfg)
    hook = _knockout_hook(sae, feature_indices)
    rows: list[dict] = []
    for language in ("en", "es"):
        evaluation = _indices(metadata, "eval", language)
        scores = clf.decision_function(scaler.transform(X[evaluation]))
        candidates = [
            index for index, score in zip(evaluation, scores)
            if labels[index] == 1 and score > 0
        ]
        selected = sorted(candidates, key=lambda index: float(clf.decision_function(scaler.transform(X[index:index + 1]))[0]), reverse=True)
        for index in selected[:cfg.examples_per_language]:
            row = metadata[index]
            original_probe = float(clf.decision_function(scaler.transform(X[index:index + 1]))[0])
            ablated_vector = X[index].copy()
            ablated_vector[feature_indices] = 0.0
            ablated_probe = float(clf.decision_function(scaler.transform(ablated_vector[None]))[0])
            original_logit = _classify(model, processor, row["image_path"], row["prompt_text"], cfg.device)
            ablated_logit = _classify(model, processor, row["image_path"], row["prompt_text"], cfg.device, target_module, hook)
            rows.append({
                "sample_id": row["sample_id"], "language": language, "gold": "sexist",
                "baseline_probe_score": original_probe, "ablated_probe_score": ablated_probe,
                "delta_probe": ablated_probe - original_probe,
                "baseline_logit": original_logit, "ablated_logit": ablated_logit,
                "delta_logit": ablated_logit - original_logit,
                "top_feature_indices": [int(value) for value in feature_indices],
            })
            logger.info("%s %s: delta_probe=%.3f delta_logit=%.3f", language, row["sample_id"], ablated_probe - original_probe, ablated_logit - original_logit)

    with (output_dir / "causal_mediation_results.jsonl").open("w") as fd:
        for row in rows:
            fd.write(json.dumps(row) + "\n")
    summary = {"top_feature_indices": [int(value) for value in feature_indices], "by_language": {}}
    for language in ("en", "es"):
        subset = [row for row in rows if row["language"] == language]
        summary["by_language"][language] = {
            "n": len(subset),
            "mean_abs_delta_probe": round(float(np.mean([abs(row["delta_probe"]) for row in subset])), 4) if subset else None,
            "mean_abs_delta_logit": round(float(np.mean([abs(row["delta_logit"]) for row in subset])), 4) if subset else None,
        }
    (output_dir / "causal_mediation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
