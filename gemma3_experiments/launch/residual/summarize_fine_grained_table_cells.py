"""Summarize the newly completed Gemma fine-grained table cells."""

import json
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import parse

from gemma3_experiments.launch.residual.probe_validate import _per_class_metrics


@dataclass
class SummaryConfig:
    native_harmeme_path: str
    hook_path: str
    hook_hierarchical_path: str
    base_sae_results_path: str
    output_path: str


SPECS = {
    "HarMeme": ("harmfulness_and_target", "harmfulness", ["not harmful", "somewhat harmful", "very harmful"]),
    "MMHS150K": ("hate_category", "label", ["NotHate", "Racist", "Sexist", "Homophobe", "Religion", "OtherHate"]),
}


def _jsonl_metrics(path: Path, dataset_name: str, *, hierarchical: bool = False) -> dict | None:
    task_name, field_name, _ = SPECS[dataset_name]
    golds, predictions = [], []
    with path.open() as fd:
        for line in fd:
            row = json.loads(line)
            if row.get("dataset_name") != dataset_name or row.get("task_name") != task_name:
                continue
            gold = (row.get("gold_fields") or {}).get(field_name)
            prediction = row.get("prediction_label")
            if gold is not None and prediction is not None:
                golds.append(gold)
                predictions.append(prediction)
    if not golds:
        return None
    metrics = _per_class_metrics(golds, predictions)
    return {"n": len(golds), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "hierarchical": hierarchical}


def main() -> None:
    cfg = parse(SummaryConfig)
    native = _jsonl_metrics(Path(cfg.native_harmeme_path), "HarMeme")
    hook_harmeme = _jsonl_metrics(Path(cfg.hook_path), "HarMeme")
    hook_mmhs = _jsonl_metrics(Path(cfg.hook_path), "MMHS150K")
    hook_mmhs_hierarchical = _jsonl_metrics(Path(cfg.hook_hierarchical_path), "MMHS150K", hierarchical=True)
    base = json.loads(Path(cfg.base_sae_results_path).read_text())
    result = {
        "Gemma::HarMeme::3class": {
            "native": native,
            "hook": hook_harmeme,
            "base_sae": base.get("HarMeme::harmfulness_and_target"),
        },
        "Gemma::MMHS150K::6class_flat": {
            "hook": hook_mmhs,
            "base_sae": base.get("MMHS150K::hate_category"),
        },
        "Gemma::MMHS150K::6class_hierarchical": {
            "native": None,
            "hook": hook_mmhs_hierarchical,
            "base_sae": base.get("MMHS150K::hate_category::hierarchical"),
        },
    }
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()