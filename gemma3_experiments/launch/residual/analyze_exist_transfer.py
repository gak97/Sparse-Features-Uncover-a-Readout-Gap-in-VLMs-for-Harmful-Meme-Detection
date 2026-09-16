

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
class ExistTransferAnalysisConfig:
    probe_predictions_path: str = ""
    baseline_predictions_path: str = ""
    feature_matrix_path: str = ""
    residual_cache_dir: str = ""
    exist_feature_rankings_path: str = ""
    mami_feature_rankings_path: str = ""
    probe_joblib_path: str = ""
    scaler_joblib_path: str = ""
    label_encoder_joblib_path: str = ""
    output_dir: str = "./exist_transfer_analysis"
    positive_label: str = "sexist"
    mami_positive_label: str = "misogynous"
    top_k_features: int = 20
    women_keywords: list[str] = ("mujer", "mujeres", "femin", "woman", "women", "girl", "female")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fd:
        return [json.loads(line) for line in fd if line.strip()]


def _load_feature_matrix(path: Path) -> tuple[list[str], scipy.sparse.csr_matrix]:
    payload = np.load(path, allow_pickle=True)
    keys = [str(key) for key in payload["keys"]]
    raw = payload["X"]
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
        raw = raw.item()
    return keys, scipy.sparse.csr_matrix(raw)


def _top_features(path: Path, label: str, top_k: int) -> list[int]:
    rankings = json.loads(path.read_text()).get("rankings", {})
    selected: list[int] = []
    for task in rankings.values():
        per_label = task.get("label_rankings", {}).get(label, {})
        for row in per_label.get("top_label_up", []):
            feature_idx = int(row["feature_idx"])
            if feature_idx not in selected:
                selected.append(feature_idx)
            if len(selected) == top_k:
                return selected
    assert selected, f"No label_up features for {label!r} in {path}."
    return selected


def _prediction_labels(clf: Any, scaler: Any, label_encoder: Any, X: scipy.sparse.csr_matrix) -> list[str]:
    X_scaled = scaler.transform(X)
    try:
        predicted = clf.predict(X_scaled)
    except TypeError:
        predicted = clf.predict(X_scaled.toarray())
    return label_encoder.inverse_transform(predicted).tolist()


def _positive_probability(clf: Any, scaler: Any, label_encoder: Any, X: scipy.sparse.csr_matrix, positive_label: str) -> np.ndarray | None:
    if not hasattr(clf, "predict_proba"):
        return None
    X_scaled = scaler.transform(X)
    try:
        probas = clf.predict_proba(X_scaled)
    except TypeError:
        probas = clf.predict_proba(X_scaled.toarray())
    class_i = list(label_encoder.classes_).index(positive_label)
    return probas[:, class_i]


def _ablate(X: scipy.sparse.csr_matrix, feature_indices: list[int]) -> scipy.sparse.csr_matrix:
    valid = sorted({feature_idx for feature_idx in feature_indices if 0 <= feature_idx < X.shape[1]})
    assert valid, "No selected features overlap the probe feature matrix."
    result = X.copy().tolil()
    result[:, valid] = 0
    return result.tocsr()


def _metrics(gold: list[str], pred: list[str], probas: np.ndarray | None, positive_label: str) -> dict[str, Any]:
    result = {
        "accuracy": round(float(accuracy_score(gold, pred)), 6),
        "macro_f1": round(float(f1_score(gold, pred, average="macro")), 6),
        "predicted_positive_rate": round(float(np.mean(np.asarray(pred) == positive_label)), 6),
    }
    if probas is not None:
        result["mean_positive_probability"] = round(float(probas.mean()), 6)
    return result


def run(cfg: ExistTransferAnalysisConfig) -> None:
    output_dpath = Path(cfg.output_dir)
    output_dpath.mkdir(parents=True, exist_ok=True)
    paths = [
        cfg.probe_predictions_path,
        cfg.feature_matrix_path,
        cfg.residual_cache_dir,
        cfg.exist_feature_rankings_path,
        cfg.mami_feature_rankings_path,
        cfg.probe_joblib_path,
        cfg.scaler_joblib_path,
        cfg.label_encoder_joblib_path,
    ]
    assert all(paths), "All analysis input paths are required."
    assert all(Path(path).exists() for path in paths), "One or more EXIST analysis inputs do not exist."

    predictions = {row["sample_key"]: row for row in _read_jsonl(Path(cfg.probe_predictions_path))}
    baseline_predictions = {}
    if cfg.baseline_predictions_path:
        baseline_predictions = {
            f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}": row
            for row in _read_jsonl(Path(cfg.baseline_predictions_path))
        }
    keys, X = _load_feature_matrix(Path(cfg.feature_matrix_path))
    assert len(keys) == X.shape[0], "Feature matrix keys and rows disagree."
    manifests = {f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}": row for row in _read_jsonl(Path(cfg.residual_cache_dir) / "manifest.jsonl")}
    rows = [(index, key, predictions[key], manifests[key]) for index, key in enumerate(keys) if key in predictions and key in manifests]
    assert rows, "No probe predictions aligned with the EXIST residual-cache manifest."

    exist_features = _top_features(Path(cfg.exist_feature_rankings_path), cfg.positive_label, cfg.top_k_features)
    mami_features = _top_features(Path(cfg.mami_feature_rankings_path), cfg.mami_positive_label, cfg.top_k_features)
    shared_features = sorted(set(exist_features) & set(mami_features))
    overlap = {
        "exist_positive_features": exist_features,
        "mami_positive_features": mami_features,
        "shared_features": shared_features,
        "n_shared": len(shared_features),
        "jaccard": round(len(shared_features) / len(set(exist_features) | set(mami_features)), 6),
    }

    fp_rows = [row for row in rows if row[2]["gold_label"] != cfg.positive_label and row[2]["probe_pred"] == cfg.positive_label]
    tn_rows = [row for row in rows if row[2]["gold_label"] != cfg.positive_label and row[2]["probe_pred"] != cfg.positive_label]
    fp_i = [row[0] for row in fp_rows]
    tn_i = [row[0] for row in tn_rows]
    fp_summary: dict[str, Any] = {"n_false_positives": len(fp_rows), "n_true_negatives": len(tn_rows)}
    if fp_i and tn_i:
        activation_delta = np.asarray(X[fp_i].mean(axis=0) - X[tn_i].mean(axis=0)).ravel()
        top_i = np.argsort(activation_delta)[::-1][: cfg.top_k_features]
        fp_summary["top_false_positive_features"] = [
            {"feature_idx": int(feature_idx), "mean_activation_delta": round(float(activation_delta[feature_idx]), 6)}
            for feature_idx in top_i
        ]
    women_keywords = tuple(keyword.lower() for keyword in cfg.women_keywords)

    def _false_positive_payload(row: tuple[int, str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
        _index, key, prediction, manifest = row
        meme_text = str(manifest.get("meme_text", ""))
        return {
            "sample_key": key,
            "sample_id": manifest["sample_id"],
            "language": manifest.get("raw_fields", {}).get("lang"),
            "meme_text": meme_text,
            "image_path": manifest.get("image_path"),
            "gold_label": prediction["gold_label"],
            "probe_pred": prediction["probe_pred"],
            "probe_proba": prediction.get("probe_proba"),
            "women_centered_text_heuristic": any(keyword in meme_text.lower() for keyword in women_keywords),
        }

    false_positive_payloads = [_false_positive_payload(row) for row in fp_rows]
    (output_dpath / "benign_false_positives.jsonl").write_text(
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in false_positive_payloads)
    )
    (output_dpath / "benign_women_centered_false_positives.jsonl").write_text(
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in false_positive_payloads if payload["women_centered_text_heuristic"])
    )

    clf = joblib.load(cfg.probe_joblib_path)
    scaler = joblib.load(cfg.scaler_joblib_path)
    label_encoder = joblib.load(cfg.label_encoder_joblib_path)
    gold = [prediction["gold_label"] for _index, _key, prediction, _manifest in rows]
    selected_i = [row[0] for row in rows]
    X_eval = X[selected_i]
    baseline_pred = _prediction_labels(clf, scaler, label_encoder, X_eval)
    baseline_proba = _positive_probability(clf, scaler, label_encoder, X_eval, cfg.positive_label)
    ablations: dict[str, Any] = {"baseline": _metrics(gold, baseline_pred, baseline_proba, cfg.positive_label)}
    for name, features in {"exist_positive": exist_features, "mami_positive": mami_features, "shared": shared_features}.items():
        if not features:
            continue
        X_ablated = _ablate(X_eval, features)
        pred = _prediction_labels(clf, scaler, label_encoder, X_ablated)
        proba = _positive_probability(clf, scaler, label_encoder, X_ablated, cfg.positive_label)
        metrics = _metrics(gold, pred, proba, cfg.positive_label)
        metrics["feature_indices"] = features
        metrics["macro_f1_delta"] = round(metrics["macro_f1"] - ablations["baseline"]["macro_f1"], 6)
        ablations[name] = metrics

    per_language_metrics: dict[str, Any] = {}
    per_language_baseline_metrics: dict[str, Any] = {}
    paired_probe_baseline: dict[str, Any] = {}
    for language in sorted({str(manifest.get("raw_fields", {}).get("lang", "unknown")) for _index, _key, _prediction, manifest in rows}):
        language_i = [index for index, (_row_i, _key, _prediction, manifest) in enumerate(rows) if str(manifest.get("raw_fields", {}).get("lang", "unknown")) == language]
        per_language_metrics[language] = _metrics(
            [gold[index] for index in language_i],
            [baseline_pred[index] for index in language_i],
            baseline_proba[language_i] if baseline_proba is not None else None,
            cfg.positive_label,
        )
        language_rows = [rows[index] for index in language_i]
        baseline_rows = [row for row in language_rows if row[1] in baseline_predictions]
        if baseline_rows:
            baseline_gold = [prediction["gold_label"] for _index, _key, prediction, _manifest in baseline_rows]
            baseline_labels = [str(baseline_predictions[key].get("prediction_label", "")) for _index, key, _prediction, _manifest in baseline_rows]
            per_language_baseline_metrics[language] = _metrics(
                baseline_gold,
                baseline_labels,
                None,
                cfg.positive_label,
            )
            paired_probe_baseline[language] = {
                "n": len(baseline_rows),
                "accuracy_delta": round(
                    per_language_metrics[language]["accuracy"] - per_language_baseline_metrics[language]["accuracy"],
                    6,
                ),
                "macro_f1_delta": round(
                    per_language_metrics[language]["macro_f1"] - per_language_baseline_metrics[language]["macro_f1"],
                    6,
                ),
            }

    report = {
        "n_aligned_samples": len(rows),
        "feature_overlap": overlap,
        "false_positive_audit": fp_summary,
        "per_language_probe_metrics": per_language_metrics,
        "per_language_baseline_metrics": per_language_baseline_metrics,
        "paired_probe_baseline": paired_probe_baseline,
        "zero_ablation": ablations,
    }
    (output_dpath / "exist_transfer_report.json").write_text(json.dumps(report, indent=2) + "\n")
    logging.info("Wrote EXIST transfer analysis to %s", output_dpath)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse(ExistTransferAnalysisConfig))


if __name__ == "__main__":
    main()