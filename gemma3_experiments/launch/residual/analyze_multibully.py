"""Audit MultiBully SAE probes with top-feature ablation and false-positive records."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy.sparse
from simple_parsing import parse
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class MultiBullyAnalysisConfig:
    probe_predictions_path: str = ""
    baseline_predictions_path: str = ""
    feature_matrix_path: str = ""
    residual_cache_dir: str = ""
    feature_rankings_path: str = ""
    probe_joblib_path: str = ""
    scaler_joblib_path: str = ""
    label_encoder_joblib_path: str = ""
    output_dir: str = "./multibully_analysis"
    positive_label: str = "Bully"
    top_k_features: int = 20


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fd:
        return [json.loads(line) for line in fd if line.strip()]


def _load_feature_matrix(path: Path) -> tuple[list[str], scipy.sparse.csr_matrix]:
    payload = np.load(path, allow_pickle=True)
    raw = payload["X"]
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
        raw = raw.item()
    return [str(key) for key in payload["keys"]], scipy.sparse.csr_matrix(raw)


def _top_features(path: Path, label: str, top_k: int) -> list[int]:
    rankings = json.loads(path.read_text()).get("rankings", {})
    selected: list[int] = []
    for task in rankings.values():
        label_ranking = task.get("label_rankings", {}).get(label, {})
        for row in label_ranking.get("top_label_up", []):
            feature_idx = int(row["feature_idx"])
            if feature_idx not in selected:
                selected.append(feature_idx)
            if len(selected) == top_k:
                return selected
    assert selected, f"No label_up features for {label!r} in {path}."
    return selected


def _metrics(gold: list[str], prediction: list[str], positive_label: str) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(gold, prediction)), 6),
        "macro_f1": round(float(f1_score(gold, prediction, average="macro")), 6),
        "predicted_positive_rate": round(float(np.mean(np.asarray(prediction) == positive_label)), 6),
    }


def _ablate(X: scipy.sparse.csr_matrix, feature_indices: list[int]) -> scipy.sparse.csr_matrix:
    valid = sorted({index for index in feature_indices if 0 <= index < X.shape[1]})
    assert valid, "No selected features overlap the probe feature matrix."
    ablated = X.copy().tolil()
    ablated[:, valid] = 0
    return ablated.tocsr()


def run(cfg: MultiBullyAnalysisConfig) -> None:
    required_paths = [
        cfg.probe_predictions_path,
        cfg.feature_matrix_path,
        cfg.residual_cache_dir,
        cfg.feature_rankings_path,
        cfg.probe_joblib_path,
        cfg.scaler_joblib_path,
        cfg.label_encoder_joblib_path,
    ]
    assert all(required_paths), "All probe, cache, ranking, and model paths are required."
    assert all(Path(path).exists() for path in required_paths), "One or more MultiBully analysis inputs do not exist."
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = {row["sample_key"]: row for row in _read_jsonl(Path(cfg.probe_predictions_path))}
    manifests = {
        f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}": row
        for row in _read_jsonl(Path(cfg.residual_cache_dir) / "manifest.jsonl")
    }
    keys, X = _load_feature_matrix(Path(cfg.feature_matrix_path))
    rows = [(index, key, predictions[key], manifests[key]) for index, key in enumerate(keys) if key in predictions and key in manifests]
    assert rows, "No MultiBully probe predictions align with the residual-cache manifest."

    feature_indices = _top_features(Path(cfg.feature_rankings_path), cfg.positive_label, cfg.top_k_features)
    gold = [prediction["gold_label"] for _index, _key, prediction, _manifest in rows]
    probe_prediction = [prediction["probe_pred"] for _index, _key, prediction, _manifest in rows]
    probe_metrics = _metrics(gold, probe_prediction, cfg.positive_label)

    fp_rows = [row for row in rows if row[2]["gold_label"] != cfg.positive_label and row[2]["probe_pred"] == cfg.positive_label]
    tn_rows = [row for row in rows if row[2]["gold_label"] != cfg.positive_label and row[2]["probe_pred"] != cfg.positive_label]
    fp_summary: dict[str, Any] = {"n_false_positives": len(fp_rows), "n_true_negatives": len(tn_rows)}
    if fp_rows and tn_rows:
        fp_i = [row[0] for row in fp_rows]
        tn_i = [row[0] for row in tn_rows]
        activation_delta = np.asarray(X[fp_i].mean(axis=0) - X[tn_i].mean(axis=0)).ravel()
        fp_summary["top_false_positive_features"] = [
            {"feature_idx": int(index), "mean_activation_delta": round(float(activation_delta[index]), 6)}
            for index in np.argsort(activation_delta)[::-1][: cfg.top_k_features]
        ]
    with (output_dir / "benign_false_positives.jsonl").open("w") as fd:
        for _index, key, prediction, manifest in fp_rows:
            fd.write(json.dumps({
                "sample_key": key,
                "sample_id": manifest["sample_id"],
                "image_path": manifest["image_path"],
                "meme_text": manifest.get("meme_text", ""),
                "gold_label": prediction["gold_label"],
                "probe_pred": prediction["probe_pred"],
                "probe_proba": prediction.get("probe_proba"),
                "source_annotations": manifest.get("raw_fields", {}),
            }, ensure_ascii=False) + "\n")

    clf = joblib.load(cfg.probe_joblib_path)
    scaler = joblib.load(cfg.scaler_joblib_path)
    label_encoder = joblib.load(cfg.label_encoder_joblib_path)
    selected_i = [row[0] for row in rows]
    X_eval = X[selected_i]
    ablated_prediction = label_encoder.inverse_transform(clf.predict(scaler.transform(_ablate(X_eval, feature_indices)))).tolist()
    ablated_metrics = _metrics(gold, ablated_prediction, cfg.positive_label)
    ablated_metrics["macro_f1_delta"] = round(ablated_metrics["macro_f1"] - probe_metrics["macro_f1"], 6)
    ablated_metrics["feature_indices"] = feature_indices

    no_hook_metrics: dict[str, Any] = {}
    paired_probe_no_hook: dict[str, Any] = {}
    if cfg.baseline_predictions_path:
        baseline_rows = {
            f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}": row
            for row in _read_jsonl(Path(cfg.baseline_predictions_path))
        }
        paired = [row for row in rows if row[1] in baseline_rows]
        if paired:
            no_hook_gold = [prediction["gold_label"] for _index, _key, prediction, _manifest in paired]
            no_hook_prediction = [str(baseline_rows[key]["prediction_label"]) for _index, key, _prediction, _manifest in paired]
            no_hook_metrics = _metrics(no_hook_gold, no_hook_prediction, cfg.positive_label)
            paired_probe_no_hook = {
                "n": len(paired),
                "accuracy_delta": round(probe_metrics["accuracy"] - no_hook_metrics["accuracy"], 6),
                "macro_f1_delta": round(probe_metrics["macro_f1"] - no_hook_metrics["macro_f1"], 6),
            }

    report = {
        "n_aligned_samples": len(rows),
        "probe_metrics": probe_metrics,
        "no_hook_metrics": no_hook_metrics,
        "paired_probe_no_hook": paired_probe_no_hook,
        "top_bully_feature_ablation": ablated_metrics,
        "false_positive_audit": fp_summary,
    }
    (output_dir / "multibully_report.json").write_text(json.dumps(report, indent=2) + "\n")
    logging.info("Wrote MultiBully analysis to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse(MultiBullyAnalysisConfig))


if __name__ == "__main__":
    main()