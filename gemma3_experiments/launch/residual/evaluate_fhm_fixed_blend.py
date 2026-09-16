"""Evaluate a fixed FHM blend without tuning on the evaluation labels."""

import argparse
import json
from pathlib import Path
from typing import Any

from gemma3_experiments.launch.residual.probe_fhm_prediction_blend import (
    LABELS,
    TASK_KEY,
    _build_pairs,
    _load_metadata,
    _load_predictions,
    _pair_metrics,
    _per_class_metrics,
)


def _blend(left: dict[str, float], right: dict[str, float], left_weight: float) -> dict[str, float]:
    return {label: left_weight * float(left[label]) + (1.0 - left_weight) * float(right[label]) for label in LABELS}


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    left_rows = _load_predictions(Path(args.left_predictions_path))
    right_rows = _load_predictions(Path(args.right_predictions_path))
    keys = [key for key in left_rows if key in right_rows]
    assert keys, "No common FHM prediction keys."
    y_gold = [left_rows[key]["gold_label"] for key in keys]
    assert all(y_gold[i] == right_rows[keys[i]]["gold_label"] for i in range(len(keys)))

    proba_by_key = {key: _blend(left_rows[key]["probe_proba"], right_rows[key]["probe_proba"], args.left_weight) for key in keys}
    pred_by_key = {key: "hateful" if proba_by_key[key]["hateful"] >= args.threshold else "not hateful" for key in keys}
    metrics = _per_class_metrics(y_gold, [pred_by_key[key] for key in keys])

    metadata = _load_metadata(Path(args.fhm_confounder_metadata_path))
    rows_by_key = {key: left_rows[key] for key in keys}
    pair_metrics = {}
    for source in args.pair_sources:
        pairs = _build_pairs(rows_by_key, metadata, source)
        pair_metrics[source] = _pair_metrics(pairs, pred_by_key, proba_by_key)

    payload: dict[str, Any] = {
        "config": vars(args),
        "task_key": TASK_KEY,
        "n": len(keys),
        "metrics": metrics,
        "pair_metrics": {source: {k: v for k, v in source_metrics.items() if k != "rows"} for source, source_metrics in pair_metrics.items()},
    }
    (output_dir / "fixed_blend_metrics.json").write_text(json.dumps(payload, indent=2))
    with (output_dir / "probe_val_predictions.jsonl").open("w") as fd:
        for key in keys:
            fd.write(json.dumps({
                "sample_key": key,
                "task_key": TASK_KEY,
                "gold_label": left_rows[key]["gold_label"],
                "probe_pred": pred_by_key[key],
                "probe_proba": {label: round(proba_by_key[key][label], 6) for label in LABELS},
                "left_probe_proba": left_rows[key]["probe_proba"],
                "right_probe_proba": right_rows[key]["probe_proba"],
                "probe_correct": pred_by_key[key] == left_rows[key]["gold_label"],
            }) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_predictions_path", required=True)
    parser.add_argument("--right_predictions_path", required=True)
    parser.add_argument("--fhm_confounder_metadata_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--left_weight", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--pair_sources", nargs="+", default=["pseudo_img", "pseudo_text"])
    run(parser.parse_args())


if __name__ == "__main__":
    main()