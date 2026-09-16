

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

LABELS = ("hateful", "not hateful")
TASK_KEY = "Hateful_Memes_Dataset::hateful_memes_binary::label"
DATASET_NAME = "Hateful_Memes_Dataset"
TASK_NAME = "hateful_memes_binary"

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fd:
        return [json.loads(line) for line in fd if line.strip()]


def _sample_key(row: dict[str, Any]) -> str:
    if row.get("sample_key"):
        return row["sample_key"]
    return f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in _read_jsonl(path):
        if row.get("task_key") != TASK_KEY:
            continue
        proba = row.get("probe_proba") or {}
        if not all(label in proba for label in LABELS):
            continue
        rows[_sample_key(row)] = row
    return rows


def _normalise_fhm_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".png") or text.endswith(".jpg") or text.endswith(".jpeg"):
        text = Path(text).stem
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalise_group_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text if text else None


def _row_sample_id(row: dict[str, Any]) -> str | None:
    key = _sample_key(row)
    return _normalise_fhm_id(key.rsplit("::", 1)[-1])


def _load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.values() if isinstance(data, dict) else data
    by_id = {}
    for row in rows:
        sample_id = _normalise_fhm_id(row.get("id") or row.get("img") or row.get("image_id"))
        if sample_id is not None:
            by_id[sample_id] = row
    return by_id


def _build_pairs(rows_by_key: dict[str, dict[str, Any]], metadata: dict[str, dict[str, Any]], source: str) -> list[tuple[str, str]]:
    source_to_col = {
        "pseudo_img": "pseudo_img_idx",
        "pseudo_image": "pseudo_img_idx",
        "pseudo_text": "pseudo_text_idx",
    }
    if source not in source_to_col:
        raise ValueError(f"Unsupported pair source: {source}")

    groups = defaultdict(list)
    for key, row in rows_by_key.items():
        sample_id = _row_sample_id(row)
        if sample_id is None:
            continue
        meta = metadata.get(sample_id)
        if meta is None:
            continue
        group_value = _normalise_group_value(meta.get(source_to_col[source]))
        if group_value is None:
            continue
        groups[group_value].append(row)

    pairs = []
    seen = set()
    for rows in groups.values():
        hateful_rows = [row for row in rows if row.get("gold_label") == "hateful"]
        benign_rows = [row for row in rows if row.get("gold_label") == "not hateful"]
        if not hateful_rows or not benign_rows:
            continue
        for hateful_row in hateful_rows:
            hateful_key = _sample_key(hateful_row)
            for benign_row in benign_rows:
                benign_key = _sample_key(benign_row)
                pair_id = tuple(sorted((hateful_key, benign_key)))
                if pair_id in seen:
                    continue
                seen.add(pair_id)
                pairs.append((hateful_key, benign_key))
    return pairs


def _per_class_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "n": len(gold),
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4),
        "macro_precision": round(sum(precisions) / len(precisions), 4),
        "macro_recall": round(sum(recalls) / len(recalls), 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "per_class": per_class,
        "gold_counts": {label: gold.count(label) for label in LABELS},
        "pred_counts": {label: pred.count(label) for label in LABELS},
    }


def _pair_metrics(pairs: list[tuple[str, str]], pred_by_key: dict[str, str], proba_by_key: dict[str, dict[str, float]]) -> dict[str, Any]:
    rows = []
    for hateful_key, benign_key in pairs:
        if hateful_key not in pred_by_key or benign_key not in pred_by_key:
            continue
        hateful_prob = proba_by_key[hateful_key]["hateful"]
        benign_prob = proba_by_key[benign_key]["hateful"]
        rows.append({
            "hateful_key": hateful_key,
            "benign_key": benign_key,
            "both_correct": pred_by_key[hateful_key] == "hateful" and pred_by_key[benign_key] == "not hateful",
            "separated": pred_by_key[hateful_key] != pred_by_key[benign_key],
            "positive_margin": hateful_prob > benign_prob,
            "prob_gap": hateful_prob - benign_prob,
        })
    if not rows:
        return {
            "n_pairs": 0,
            "both_correct_rate": 0.0,
            "separated_rate": 0.0,
            "positive_margin_rate": 0.0,
            "median_prob_gap": 0.0,
            "rows": [],
        }
    return {
        "n_pairs": len(rows),
        "both_correct_rate": round(sum(row["both_correct"] for row in rows) / len(rows), 6),
        "separated_rate": round(sum(row["separated"] for row in rows) / len(rows), 6),
        "positive_margin_rate": round(sum(row["positive_margin"] for row in rows) / len(rows), 6),
        "median_prob_gap": round(median(row["prob_gap"] for row in rows), 6),
        "rows": rows,
    }


def _blend_proba(left: dict[str, float], right: dict[str, float], left_weight: float) -> dict[str, float]:
    return {
        label: left_weight * float(left[label]) + (1.0 - left_weight) * float(right[label])
        for label in LABELS
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    left_rows = _load_predictions(Path(args.layer22_predictions_path))
    right_rows = _load_predictions(Path(args.layer29_predictions_path))
    keys = [key for key in left_rows if key in right_rows]
    assert keys, "No common FHM prediction keys."
    logger.info("Common FHM prediction keys: %d", len(keys))

    y_gold = [left_rows[key]["gold_label"] for key in keys]
    assert all(y_gold[i] == right_rows[keys[i]]["gold_label"] for i in range(len(keys)))
    sweep = []
    best = None
    for w_int in range(args.weight_steps + 1):
        layer22_weight = w_int / args.weight_steps
        proba_by_key = {
            key: _blend_proba(left_rows[key]["probe_proba"], right_rows[key]["probe_proba"], layer22_weight)
            for key in keys
        }
        for t_int in range(args.threshold_min, args.threshold_max + 1):
            threshold = t_int / 100.0
            pred = ["hateful" if proba_by_key[key]["hateful"] >= threshold else "not hateful" for key in keys]
            metrics = _per_class_metrics(y_gold, pred)
            row = {
                "layer22_weight": round(layer22_weight, 4),
                "layer29_weight": round(1.0 - layer22_weight, 4),
                "threshold": round(threshold, 4),
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "hateful_precision": metrics["per_class"]["hateful"]["precision"],
                "hateful_recall": metrics["per_class"]["hateful"]["recall"],
                "pred_hateful": metrics["pred_counts"]["hateful"],
            }
            sweep.append(row)
            if best is None or row["macro_f1"] > best["macro_f1"]:
                best = row

    assert best is not None
    best_proba_by_key = {
        key: _blend_proba(left_rows[key]["probe_proba"], right_rows[key]["probe_proba"], best["layer22_weight"])
        for key in keys
    }
    best_pred_by_key = {
        key: "hateful" if best_proba_by_key[key]["hateful"] >= best["threshold"] else "not hateful"
        for key in keys
    }
    best_metrics = _per_class_metrics(y_gold, [best_pred_by_key[key] for key in keys])
    rows_by_key = {key: left_rows[key] for key in keys}
    metadata = _load_metadata(Path(args.fhm_confounder_metadata_path))
    pair_metrics = {}
    for source in args.pair_sources:
        pairs = _build_pairs(rows_by_key, metadata, source)
        pair_metrics[source] = _pair_metrics(pairs, best_pred_by_key, best_proba_by_key)

    metrics = {
        "config": vars(args),
        "n_val": len(keys),
        "best": best,
        "best_metrics": best_metrics,
        "top_sweep": sorted(sweep, key=lambda row: row["macro_f1"], reverse=True)[:25],
        "pair_metrics": {
            source: {key: value for key, value in source_metrics.items() if key != "rows"}
            for source, source_metrics in pair_metrics.items()
        },
    }
    (output_dir / "prediction_blend_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (output_dir / "probe_val_predictions.jsonl").open("w") as fd:
        for key in keys:
            fd.write(json.dumps({
                "sample_key": key,
                "task_key": TASK_KEY,
                "gold_label": left_rows[key]["gold_label"],
                "probe_pred": best_pred_by_key[key],
                "probe_proba": {label: round(best_proba_by_key[key][label], 6) for label in LABELS},
                "layer22_probe_proba": left_rows[key]["probe_proba"],
                "layer29_probe_proba": right_rows[key]["probe_proba"],
                "probe_correct": best_pred_by_key[key] == left_rows[key]["gold_label"],
            }) + "\n")
    with (output_dir / "validation_pair_metrics.jsonl").open("w") as fd:
        for source, source_metrics in pair_metrics.items():
            for row in source_metrics.get("rows", []):
                fd.write(json.dumps({"source": source, **row}) + "\n")

    logger.info(
        "FHM prediction blend: macro_f1=%.4f acc=%.4f w22=%.2f threshold=%.2f",
        best_metrics["macro_f1"],
        best_metrics["accuracy"],
        best["layer22_weight"],
        best["threshold"],
    )
    logger.info("Wrote outputs to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer22_predictions_path", required=True)
    parser.add_argument("--layer29_predictions_path", required=True)
    parser.add_argument("--fhm_confounder_metadata_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pair_sources", nargs="+", default=["pseudo_img", "pseudo_text"])
    parser.add_argument("--weight_steps", type=int, default=100)
    parser.add_argument("--threshold_min", type=int, default=1)
    parser.add_argument("--threshold_max", type=int, default=99)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
