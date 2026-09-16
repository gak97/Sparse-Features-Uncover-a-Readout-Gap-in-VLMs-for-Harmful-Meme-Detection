"""Reroute a MultiBully base-SAE probe score into Qwen's output decision."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simple_parsing import Serializable, list_field, parse

from ..qwen import QwenSae
from ..task_labels import fit_scaled_logreg
from .analyze_exist_multilingual import _extract_features, _read_jsonl
from .analyze_multibully import _metrics


@dataclass
class MultiBullyLogitProjectionConfig(Serializable):
    cache_dir: str = "./outputs/multibully/dense_raw"
    predictions_path: str = "./outputs/multibully/baseline_raw_ocr/baseline_eval.jsonl"
    output_dir: str = "./outputs/multibully/logit_projection_raw_ocr"
    device: str = "cuda:0"
    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    C: float = 1.0
    max_iter: int = 3000
    calibration_fraction: float = 0.30
    seed: int = 20260803
    alpha_grid: list[float] = list_field(0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1 - 1e-6)
    return math.log(probability / (1 - probability))


def _stratified_calibration(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for label in (0, 1):
        group = np.flatnonzero(labels == label)
        rng.shuffle(group)
        indices.extend(group[:max(1, round(len(group) * fraction))])
    return np.asarray(sorted(indices), dtype=np.int64)


def _select_alpha(logits: np.ndarray, probe_z: np.ndarray, labels: np.ndarray, grid: list[float]) -> float:
    best_alpha, best_f1 = 0.0, -1.0
    for alpha in grid:
        metric = _metrics(labels, (logits + alpha * probe_z > 0).astype(np.int32))
        if metric["macro_f1"] > best_f1:
            best_alpha, best_f1 = alpha, metric["macro_f1"]
    return best_alpha


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(MultiBullyLogitProjectionConfig)
    cache_dir, output_dir = Path(cfg.cache_dir), Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [row for row in _read_jsonl(cache_dir / "manifest.jsonl") if row.get("task_name") == "cyberbullying_detection"]
    sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    metadata, features = _extract_features(sae, cache_dir, manifest, cfg.device)
    labels = np.asarray([int(row["gold_fields"]["label"] == "bully") for row in metadata], dtype=np.int32)
    train = np.asarray([index for index, row in enumerate(metadata) if row["split"] == "probe_train"], dtype=np.int64)
    evaluation = np.asarray([index for index, row in enumerate(metadata) if row["split"] == "eval"], dtype=np.int64)
    classifier, scaler = fit_scaled_logreg(features["all"][train], labels[train], C=cfg.C, max_iter=cfg.max_iter)
    probe_score = classifier.decision_function(scaler.transform(features["all"][evaluation]))
    eval_labels = labels[evaluation]

    predictions = {str(row["sample_id"]): row for row in _read_jsonl(Path(cfg.predictions_path))}
    eval_meta = [metadata[index] for index in evaluation]
    if {str(row["sample_id"]) for row in eval_meta} != set(predictions):
        raise ValueError("Baseline prediction IDs do not match the locked MultiBully evaluation manifest")
    logits = np.asarray([_logit(predictions[str(row["sample_id"])]["conf_gap"]) for row in eval_meta])
    calibration = _stratified_calibration(eval_labels, cfg.calibration_fraction, cfg.seed)
    test_mask = np.ones(len(evaluation), dtype=bool)
    test_mask[calibration] = False
    test = np.flatnonzero(test_mask)
    calibration_mean = float(probe_score[calibration].mean())
    calibration_std = max(float(probe_score[calibration].std()), 1e-8)
    probe_z = (probe_score - calibration_mean) / calibration_std
    alpha = _select_alpha(logits[calibration], probe_z[calibration], eval_labels[calibration], cfg.alpha_grid)

    baseline = (logits[test] > 0).astype(np.int32)
    probe = (probe_z[test] > 0).astype(np.int32)
    combined = (logits[test] + alpha * probe_z[test] > 0).astype(np.int32)
    output = {
        "alpha": alpha,
        "alpha_grid": cfg.alpha_grid,
        "calibration_n": int(len(calibration)),
        "test_n": int(len(test)),
        "probe_zscore_calibration_mean": calibration_mean,
        "probe_zscore_calibration_std": calibration_std,
        "baseline": _metrics(eval_labels[test], baseline),
        "probe_only": _metrics(eval_labels[test], probe),
        "combined": _metrics(eval_labels[test], combined),
    }
    (output_dir / "logit_projection_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    logging.info("Wrote MultiBully direct-logit projection results to %s", output_dir)


if __name__ == "__main__":
    main()
