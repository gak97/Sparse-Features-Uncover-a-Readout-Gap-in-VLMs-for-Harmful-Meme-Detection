

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

LABELS = ("hateful", "not hateful")
TASK_KEY = "Hateful_Memes_Dataset::hateful_memes_binary::label"

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fd:
        return [json.loads(line) for line in fd if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as fd:
        for row in rows:
            fd.write(json.dumps(row) + "\n")


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
    if text.endswith((".png", ".jpg", ".jpeg")):
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


def _load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = data.values() if isinstance(data, dict) else data
    by_id = {}
    for row in rows:
        sample_id = row.get("id")
        if sample_id is None and row.get("img"):
            sample_id = Path(str(row["img"])).stem
        sample_id = _normalise_fhm_id(sample_id)
        if sample_id is not None:
            by_id[sample_id] = row
    return by_id


def _sample_id_from_key(key: str) -> str | None:
    return _normalise_fhm_id(key.rsplit("::", maxsplit=1)[-1])


def _metadata_for_key(key: str, metadata: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    sample_id = _sample_id_from_key(key)
    if sample_id is None:
        return None
    return metadata.get(sample_id)


def _build_pair_membership(keys: list[str], rows_by_key: dict[str, dict[str, Any]], metadata: dict[str, dict[str, Any]], source: str) -> tuple[set[str], list[tuple[str, str]]]:
    source_to_col = {
        "pseudo_img": "pseudo_img_idx",
        "pseudo_image": "pseudo_img_idx",
        "pseudo_text": "pseudo_text_idx",
    }
    if source not in source_to_col:
        raise ValueError(f"Unsupported pair source: {source}")

    groups = defaultdict(list)
    for key in keys:
        meta = _metadata_for_key(key, metadata)
        if meta is None:
            continue
        group_value = _normalise_group_value(meta.get(source_to_col[source]))
        if group_value is None:
            continue
        groups[group_value].append(key)

    seen = set()
    pair_keys = set()
    pairs = []
    for group_keys in groups.values():
        hateful_keys = [key for key in group_keys if rows_by_key[key]["gold_label"] == "hateful"]
        benign_keys = [key for key in group_keys if rows_by_key[key]["gold_label"] == "not hateful"]
        for hateful_key in hateful_keys:
            for benign_key in benign_keys:
                pair_id = tuple(sorted((hateful_key, benign_key)))
                if pair_id in seen:
                    continue
                seen.add(pair_id)
                pairs.append((hateful_key, benign_key))
                pair_keys.update((hateful_key, benign_key))
    return pair_keys, pairs


def _pred_from_threshold(proba: dict[str, float], threshold: float) -> str:
    return "hateful" if float(proba["hateful"]) >= threshold else "not hateful"


def _pred_from_negative_threshold(proba: dict[str, float], threshold: float) -> str:
    return "hateful" if float(proba["not hateful"]) < threshold else "not hateful"


def _blend_proba(left: dict[str, float], right: dict[str, float], left_weight: float) -> dict[str, float]:
    return {
        label: left_weight * float(left[label]) + (1.0 - left_weight) * float(right[label])
        for label in LABELS
    }


def _margin(proba: dict[str, float], threshold: float) -> float:
    return float(proba["hateful"]) - threshold


def _confidence(proba: dict[str, float], threshold: float) -> float:
    return abs(_margin(proba, threshold))


def _negative_margin(proba: dict[str, float], threshold: float) -> float:
    return threshold - float(proba["not hateful"])


def _negative_confidence(proba: dict[str, float], threshold: float) -> float:
    return abs(_negative_margin(proba, threshold))


def _entropy(proba: dict[str, float]) -> float:
    total = 0.0
    for label in LABELS:
        p = min(max(float(proba[label]), 1e-12), 1.0)
        total -= p * math.log(p)
    return total


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
        "gold_counts": dict(Counter(gold)),
        "pred_counts": dict(Counter(pred)),
    }


def _summarise_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _slice_metrics(rows: list[dict[str, Any]], pred_field: str, group_field: str) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field))].append(row)
    out = []
    for value, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        gold = [row["gold_label"] for row in group_rows]
        pred = [row[pred_field] for row in group_rows]
        metrics = _per_class_metrics(gold, pred)
        out.append({"field": group_field, "value": value, **metrics})
    return out


def _selector_metrics(rows: list[dict[str, Any]], name: str, chooser: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    gold = [row["gold_label"] for row in rows]
    pred = [chooser(row) for row in rows]
    metrics = _per_class_metrics(gold, pred)
    return {"selector": name, **metrics}


def _threshold_selector_search(rows: list[dict[str, Any]], signal_field: str) -> dict[str, Any]:
    candidates = sorted({float(row[signal_field]) for row in rows})
    if not candidates:
        return {"signal": signal_field, "best": None, "top": []}
    thresholds = candidates
    if len(thresholds) > 200:
        thresholds = [candidates[int(i * (len(candidates) - 1) / 199)] for i in range(200)]
    scored = []
    gold = [row["gold_label"] for row in rows]
    for threshold in thresholds:
        pred_high = [row["l22_cal_pred"] if float(row[signal_field]) >= threshold else row["l29_cal_pred"] for row in rows]
        pred_low = [row["l22_cal_pred"] if float(row[signal_field]) <= threshold else row["l29_cal_pred"] for row in rows]
        high_metrics = _per_class_metrics(gold, pred_high)
        low_metrics = _per_class_metrics(gold, pred_low)
        scored.append({
            "signal": signal_field,
            "rule": "choose_l22_if_signal_ge_threshold",
            "threshold": round(threshold, 6),
            "macro_f1": high_metrics["macro_f1"],
            "accuracy": high_metrics["accuracy"],
        })
        scored.append({
            "signal": signal_field,
            "rule": "choose_l22_if_signal_le_threshold",
            "threshold": round(threshold, 6),
            "macro_f1": low_metrics["macro_f1"],
            "accuracy": low_metrics["accuracy"],
        })
    scored.sort(key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)
    return {"signal": signal_field, "best": scored[0], "top": scored[:10]}


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    l22_rows = _load_predictions(Path(args.layer22_predictions_path))
    l29_rows = _load_predictions(Path(args.layer29_predictions_path))
    blend_rows = _load_predictions(Path(args.blend_predictions_path)) if args.blend_predictions_path else {}
    keys = [key for key in l22_rows if key in l29_rows]
    assert keys, "No common FHM prediction keys across layer-22 and layer-29."
    logger.info("Common FHM prediction keys: %d", len(keys))

    metadata = _load_metadata(Path(args.fhm_confounder_metadata_path))
    pair_key_by_source = {}
    pair_count_by_source = {}
    for source in args.pair_sources:
        pair_keys, pairs = _build_pair_membership(keys, l22_rows, metadata, source)
        pair_key_by_source[source] = pair_keys
        pair_count_by_source[source] = len(pairs)

    rows = []
    for key in keys:
        l22 = l22_rows[key]
        l29 = l29_rows[key]
        gold = l22["gold_label"]
        assert gold == l29["gold_label"]
        l22_proba = l22["probe_proba"]
        l29_proba = l29["probe_proba"]
        blend_proba = _blend_proba(l22_proba, l29_proba, args.layer22_blend_weight)
        blend_pred = _pred_from_threshold(blend_proba, args.blend_threshold)
        if key in blend_rows:
            blend_pred = blend_rows[key].get("probe_pred", blend_pred)
            blend_proba = blend_rows[key].get("probe_proba", blend_proba)
        l22_cal_pred = _pred_from_negative_threshold(l22_proba, args.layer22_threshold)
        l29_cal_pred = _pred_from_negative_threshold(l29_proba, args.layer29_threshold)
        baseline_pred = l22.get("baseline_pred") or l29.get("baseline_pred")
        source_membership = {f"in_{source}_pair": key in pair_key_by_source[source] for source in args.pair_sources}
        row = {
            "sample_key": key,
            "gold_label": gold,
            "baseline_pred": baseline_pred,
            "baseline_correct": baseline_pred == gold if baseline_pred in LABELS else None,
            "l22_raw_pred": l22.get("probe_pred"),
            "l29_raw_pred": l29.get("probe_pred"),
            "l22_cal_pred": l22_cal_pred,
            "l29_cal_pred": l29_cal_pred,
            "blend_pred": blend_pred,
            "l22_p_hateful": float(l22_proba["hateful"]),
            "l29_p_hateful": float(l29_proba["hateful"]),
            "blend_p_hateful": float(blend_proba["hateful"]),
            "l22_margin": _negative_margin(l22_proba, args.layer22_threshold),
            "l29_margin": _negative_margin(l29_proba, args.layer29_threshold),
            "blend_margin": _margin(blend_proba, args.blend_threshold),
            "l22_confidence": _negative_confidence(l22_proba, args.layer22_threshold),
            "l29_confidence": _negative_confidence(l29_proba, args.layer29_threshold),
            "blend_confidence": _confidence(blend_proba, args.blend_threshold),
            "l22_entropy": _entropy(l22_proba),
            "l29_entropy": _entropy(l29_proba),
            "confidence_diff_l22_minus_l29": _confidence(l22_proba, args.layer22_threshold) - _confidence(l29_proba, args.layer29_threshold),
            "p_hateful_diff_l22_minus_l29": float(l22_proba["hateful"]) - float(l29_proba["hateful"]),
            "l22_l29_agree_raw": l22.get("probe_pred") == l29.get("probe_pred"),
            "l22_l29_agree_cal": l22_cal_pred == l29_cal_pred,
            "l22_raw_correct": l22.get("probe_pred") == gold,
            "l29_raw_correct": l29.get("probe_pred") == gold,
            "l22_cal_correct": l22_cal_pred == gold,
            "l29_cal_correct": l29_cal_pred == gold,
            "blend_correct": blend_pred == gold,
            **source_membership,
        }
        if row["l22_raw_correct"] and row["l29_raw_correct"]:
            row["raw_complementarity"] = "both_right"
        elif row["l22_raw_correct"]:
            row["raw_complementarity"] = "l22_only"
        elif row["l29_raw_correct"]:
            row["raw_complementarity"] = "l29_only"
        else:
            row["raw_complementarity"] = "both_wrong"
        if row["l22_cal_correct"] and row["l29_cal_correct"]:
            row["cal_complementarity"] = "both_right"
        elif row["l22_cal_correct"]:
            row["cal_complementarity"] = "l22_only"
        elif row["l29_cal_correct"]:
            row["cal_complementarity"] = "l29_only"
        else:
            row["cal_complementarity"] = "both_wrong"
        rows.append(row)

    selector_rows = [
        _selector_metrics(rows, "layer22_calibrated", lambda row: row["l22_cal_pred"]),
        _selector_metrics(rows, "layer29_calibrated", lambda row: row["l29_cal_pred"]),
        _selector_metrics(rows, "frozen_prediction_blend", lambda row: row["blend_pred"]),
        _selector_metrics(rows, "choose_higher_calibrated_confidence", lambda row: row["l22_cal_pred"] if row["l22_confidence"] >= row["l29_confidence"] else row["l29_cal_pred"]),
        _selector_metrics(rows, "choose_lower_entropy", lambda row: row["l22_cal_pred"] if row["l22_entropy"] <= row["l29_entropy"] else row["l29_cal_pred"]),
        _selector_metrics(rows, "agree_else_layer22", lambda row: row["l22_cal_pred"] if row["l22_cal_pred"] == row["l29_cal_pred"] else row["l22_cal_pred"]),
        _selector_metrics(rows, "agree_else_layer29", lambda row: row["l22_cal_pred"] if row["l22_cal_pred"] == row["l29_cal_pred"] else row["l29_cal_pred"]),
        _selector_metrics(rows, "baseline_when_available_else_blend", lambda row: row["baseline_pred"] if row["baseline_pred"] in LABELS else row["blend_pred"]),
    ]
    oracle_pred = []
    for row in rows:
        if row["l22_cal_correct"]:
            oracle_pred.append(row["l22_cal_pred"])
        elif row["l29_cal_correct"]:
            oracle_pred.append(row["l29_cal_pred"])
        else:
            oracle_pred.append(row["blend_pred"])
    selector_rows.append({"selector": "oracle_l22_or_l29_calibrated", **_per_class_metrics([row["gold_label"] for row in rows], oracle_pred)})
    selector_rows.sort(key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)

    slice_rows = []
    slice_fields = ["raw_complementarity", "cal_complementarity", "l22_l29_agree_cal", "baseline_correct"]
    slice_fields.extend([f"in_{source}_pair" for source in args.pair_sources])
    for field in slice_fields:
        slice_rows.extend(_slice_metrics(rows, "blend_pred", field))

    category_rows = []
    for category, category_rows_raw in sorted(defaultdict(list, {cat: [row for row in rows if row["raw_complementarity"] == cat] for cat in sorted({row["raw_complementarity"] for row in rows})}).items()):
        category_rows.append({
            "category": category,
            "n": len(category_rows_raw),
            "l22_confidence": _summarise_numeric([row["l22_confidence"] for row in category_rows_raw]),
            "l29_confidence": _summarise_numeric([row["l29_confidence"] for row in category_rows_raw]),
            "confidence_diff_l22_minus_l29": _summarise_numeric([row["confidence_diff_l22_minus_l29"] for row in category_rows_raw]),
            "p_hateful_diff_l22_minus_l29": _summarise_numeric([row["p_hateful_diff_l22_minus_l29"] for row in category_rows_raw]),
            "baseline_correct_rate": round(sum(row["baseline_correct"] is True for row in category_rows_raw) / len(category_rows_raw), 6) if category_rows_raw else 0.0,
            **{f"{source}_pair_rate": round(sum(row[f"in_{source}_pair"] for row in category_rows_raw) / len(category_rows_raw), 6) if category_rows_raw else 0.0 for source in args.pair_sources},
        })

    disagreement_rows = [row for row in rows if row["l22_cal_pred"] != row["l29_cal_pred"]]
    threshold_search = [
        _threshold_selector_search(disagreement_rows, signal)
        for signal in [
            "confidence_diff_l22_minus_l29",
            "p_hateful_diff_l22_minus_l29",
            "l22_confidence",
            "l29_confidence",
            "l22_p_hateful",
            "l29_p_hateful",
        ]
    ]

    summary = {
        "config": vars(args),
        "n": len(rows),
        "pair_counts": pair_count_by_source,
        "metrics": {
            "layer22_raw": _per_class_metrics([row["gold_label"] for row in rows], [row["l22_raw_pred"] for row in rows]),
            "layer29_raw": _per_class_metrics([row["gold_label"] for row in rows], [row["l29_raw_pred"] for row in rows]),
            "layer22_calibrated": _per_class_metrics([row["gold_label"] for row in rows], [row["l22_cal_pred"] for row in rows]),
            "layer29_calibrated": _per_class_metrics([row["gold_label"] for row in rows], [row["l29_cal_pred"] for row in rows]),
            "blend": _per_class_metrics([row["gold_label"] for row in rows], [row["blend_pred"] for row in rows]),
        },
        "raw_complementarity_counts": dict(Counter(row["raw_complementarity"] for row in rows)),
        "calibrated_complementarity_counts": dict(Counter(row["cal_complementarity"] for row in rows)),
        "selector_metrics": selector_rows,
        "slice_metrics_for_blend": slice_rows,
        "raw_complementarity_feature_summary": category_rows,
        "disagreement_threshold_search": threshold_search,
    }
    (output_dir / "selector_error_analysis.json").write_text(json.dumps(summary, indent=2))
    _write_jsonl(output_dir / "per_sample_selector_analysis.jsonl", rows)
    _write_jsonl(output_dir / "selector_metrics.jsonl", selector_rows)
    _write_jsonl(output_dir / "slice_metrics_for_blend.jsonl", slice_rows)
    _write_jsonl(output_dir / "raw_complementarity_feature_summary.jsonl", category_rows)

    logger.info("Wrote selector/error analysis to %s", output_dir)
    logger.info("Best selector: %s macro_f1=%.4f acc=%.4f", selector_rows[0]["selector"], selector_rows[0]["macro_f1"], selector_rows[0]["accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer22_predictions_path", required=True)
    parser.add_argument("--layer29_predictions_path", required=True)
    parser.add_argument("--fhm_confounder_metadata_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--blend_predictions_path", default="")
    parser.add_argument("--pair_sources", nargs="+", default=["pseudo_img", "pseudo_text"])
    parser.add_argument("--layer22_threshold", type=float, default=0.75)
    parser.add_argument("--layer29_threshold", type=float, default=0.70)
    parser.add_argument("--layer22_blend_weight", type=float, default=0.28)
    parser.add_argument("--blend_threshold", type=float, default=0.29)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args)


if __name__ == "__main__":
    main()
