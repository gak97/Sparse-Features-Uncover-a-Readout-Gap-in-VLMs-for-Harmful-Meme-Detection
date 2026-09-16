import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert rows, f"No prediction rows found in {path}"
    return rows


def _primary_labels(row: dict[str, Any]) -> tuple[str, ...]:
    labels = row["schema"]["primary_field"]["labels"]
    return tuple(str(label) for label in labels)


def _gold_label(row: dict[str, Any]) -> str:
    field_name = row["schema"]["primary_field"]["name"]
    gold = row["gold_fields"][field_name]
    assert isinstance(gold, str), f"Expected scalar string gold label for {row['dataset_name']}:{row['sample_id']}, got {gold!r}"
    return gold


def _prediction_label(row: dict[str, Any]) -> str:
    pred = row.get("prediction_label", row.get("prediction_text"))
    assert isinstance(pred, str), f"Expected string prediction for {row['dataset_name']}:{row['sample_id']}, got {pred!r}"
    return pred


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset_name"]].append(row)

    summary: dict[str, Any] = {"overall": _summarize_group(rows)}
    summary["datasets"] = {dataset: _summarize_group(dataset_rows) for dataset, dataset_rows in sorted(by_dataset.items())}
    return summary


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    pred_counts: Counter[str] = Counter()
    gold_counts: Counter[str] = Counter()
    binary_rows: list[tuple[str, str, str]] = []
    per_class_tp: Counter[str] = Counter()
    per_class_fp: Counter[str] = Counter()
    per_class_fn: Counter[str] = Counter()
    for row in rows:
        gold = _gold_label(row)
        pred = _prediction_label(row)
        labels = _primary_labels(row)
        correct += int(pred == gold)
        pred_counts[pred] += 1
        gold_counts[gold] += 1
        if pred == gold:
            per_class_tp[gold] += 1
        else:
            per_class_fp[pred] += 1
            per_class_fn[gold] += 1
        if len(labels) == 2:
            binary_rows.append((gold, pred, labels[-1]))

    all_labels = sorted(set(gold_counts) | set(pred_counts))
    per_class: dict[str, Any] = {}
    macro_p = macro_r = macro_f1 = 0.0
    for lbl in all_labels:
        tp = per_class_tp[lbl]
        fp = per_class_fp[lbl]
        fn = per_class_fn[lbl]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_class[lbl] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}
        macro_p += p
        macro_r += r
        macro_f1 += f
    n_labels = len(all_labels)

    result: dict[str, Any] = {
        "n": len(rows),
        "accuracy": correct / len(rows),
        "gold_counts": dict(sorted(gold_counts.items())),
        "prediction_counts": dict(sorted(pred_counts.items())),
        "per_class": per_class,
        "macro_precision": round(macro_p / n_labels, 4) if n_labels else 0.0,
        "macro_recall": round(macro_r / n_labels, 4) if n_labels else 0.0,
        "macro_f1": round(macro_f1 / n_labels, 4) if n_labels else 0.0,
    }
    if binary_rows:
        tp = fp = fn = tn = 0
        for gold, pred, positive_label in binary_rows:
            gold_pos = gold == positive_label
            pred_pos = pred == positive_label
            tp += int(gold_pos and pred_pos)
            fp += int((not gold_pos) and pred_pos)
            fn += int(gold_pos and not pred_pos)
            tn += int((not gold_pos) and not pred_pos)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.update(
            {
                "binary_n": len(binary_rows),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_positive_rate": (tp + fp) / len(binary_rows),
                "gold_positive_rate": (tp + fn) / len(binary_rows),
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            }
        )
    return result


def _paired_summary(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_by_key = {
        (row["dataset_name"], row["task_name"], row["split"], row["sample_id"]): row
        for row in baseline_rows
    }
    pairs = []
    for row in rows:
        key = (row["dataset_name"], row["task_name"], row["split"], row["sample_id"])
        if key in baseline_by_key:
            pairs.append((row, baseline_by_key[key]))
    assert pairs, "No paired predictions found against baseline"

    model_correct = baseline_correct = same_prediction = 0
    for row, baseline in pairs:
        gold = _gold_label(row)
        pred = _prediction_label(row)
        baseline_pred = _prediction_label(baseline)
        model_correct += int(pred == gold)
        baseline_correct += int(baseline_pred == gold)
        same_prediction += int(pred == baseline_pred)
    n = len(pairs)
    return {
        "paired_n": n,
        "accuracy": model_correct / n,
        "baseline_accuracy": baseline_correct / n,
        "delta_accuracy": (model_correct - baseline_correct) / n,
        "same_prediction_rate": same_prediction / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--compare_to")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.predictions))
    summary = _summarize(rows)
    if args.compare_to:
        summary["paired_baseline"] = _paired_summary(rows, _read_jsonl(Path(args.compare_to)))

    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
