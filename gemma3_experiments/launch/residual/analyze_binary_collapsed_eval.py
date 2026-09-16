

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.probe_validate import _per_class_metrics

logger = logging.getLogger(__name__)


@dataclass
class BinaryCollapsedEvalConfig:
    manifest_path: str = ""       # lora_distillation_targets_all/distillation_manifest.json
    baseline_eval_dir: str = ""   # lora_distillation_eval_all_baseline
    lora_eval_dir: str = ""       # lora_distillation_eval_all_lora
    output_dir: str = "./lora_distillation_binary_collapsed_eval"


def _sanitize_task_key(task_key: str) -> str:
    return task_key.replace("::", "__").replace(" ", "_")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collapse_label(label: str, negative_labels: list[str], positive_label: str) -> str:
    return label if label in negative_labels else positive_label


def _collapsed_metrics(rows: list[dict[str, Any]], negative_labels: list[str], positive_label: str) -> dict[str, Any]:
    gold = [_collapse_label(r["gold"], negative_labels, positive_label) for r in rows]
    pred = [_collapse_label(r["pred"], negative_labels, positive_label) for r in rows]
    return _per_class_metrics(gold, pred)


def run(cfg: BinaryCollapsedEvalConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(cfg.manifest_path).read_text())
    multiclass_tasks = [entry for entry in manifest if len(entry["labels"]) > 2]
    assert multiclass_tasks, f"No multi-class tasks (>2 labels) found in {cfg.manifest_path}"

    results: dict[str, Any] = {}
    for entry in multiclass_tasks:
        task_key = entry["task_key"]
        negative_labels = entry.get("collapse_negative_labels")
        positive_label = entry.get("semantic_positive_label")

        if not negative_labels or not positive_label:
            results[task_key] = {
                "applicable": False,
                "reason": (
                    "No natural binary collapse defined for this task "
                    f"(labels={entry['labels']}) -- not a hate/not-hate style classification."
                ),
            }
            logger.info("%s: N/A (no binary collapse defined)", task_key)
            continue

        sanitized = _sanitize_task_key(task_key)
        baseline_rows = _read_jsonl(Path(cfg.baseline_eval_dir) / f"predictions_{sanitized}.jsonl")
        lora_rows = _read_jsonl(Path(cfg.lora_eval_dir) / f"predictions_{sanitized}.jsonl")
        assert baseline_rows, f"No baseline predictions found for {task_key} at {cfg.baseline_eval_dir}"
        assert lora_rows, f"No LoRA predictions found for {task_key} at {cfg.lora_eval_dir}"

        baseline_metrics = _collapsed_metrics(baseline_rows, negative_labels, positive_label)
        lora_metrics = _collapsed_metrics(lora_rows, negative_labels, positive_label)
        delta = round(lora_metrics["macro_f1"] - baseline_metrics["macro_f1"], 4)

        results[task_key] = {
            "applicable": True,
            "positive_label": positive_label,
            "negative_labels": negative_labels,
            "raw_labels": entry["labels"],
            "baseline": baseline_metrics,
            "lora": lora_metrics,
            "delta_macro_f1": delta,
        }
        logger.info(
            "%s [binary collapse %s vs %s]: baseline macro_f1=%.4f lora macro_f1=%.4f delta=%+.4f",
            task_key, positive_label, negative_labels, baseline_metrics["macro_f1"], lora_metrics["macro_f1"], delta,
        )

    (out_dir / "binary_collapsed_metrics.json").write_text(json.dumps(results, indent=2))
    logger.info("Wrote binary_collapsed_metrics.json to %s", out_dir)

    print("\n" + "=" * 100)
    print("Binary-collapsed evaluation (multi-class tasks only)")
    print(
        f"{'task_key':55s} {'n':>6s} {'base_f1':>9s} {'lora_f1':>9s} {'delta':>8s} "
        f"{'base_pos%':>10s} {'lora_pos%':>10s} {'gold_pos%':>10s}"
    )
    for task_key, result in results.items():
        if not result["applicable"]:
            print(f"{task_key:55s}  N/A -- {result['reason']}")
            continue
        b, l = result["baseline"], result["lora"]
        pos_label = result["positive_label"]
        b_pos_pct = 100.0 * b["pred_counts"].get(pos_label, 0) / b["n"]
        l_pos_pct = 100.0 * l["pred_counts"].get(pos_label, 0) / l["n"]
        gold_pos_pct = 100.0 * b["gold_counts"].get(pos_label, 0) / b["n"]
        print(
            f"{task_key:55s} {b['n']:>6d} {b['macro_f1']:>9.4f} {l['macro_f1']:>9.4f} "
            f"{result['delta_macro_f1']:>+8.4f} {b_pos_pct:>9.1f}% {l_pos_pct:>9.1f}% {gold_pos_pct:>9.1f}%"
        )
    print("=" * 100)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(BinaryCollapsedEvalConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
