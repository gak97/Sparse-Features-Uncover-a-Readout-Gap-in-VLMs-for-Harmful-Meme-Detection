

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.reward_monitor import (
    IDENTITY_GROUP_TERMS,
    contains_identity_group_term,
)
from gemma3_experiments.residual_sae import MemeDatasetConfig, load_meme_samples

logger = logging.getLogger(__name__)


@dataclass
class BuildIdentityGroupControlsConfig:
    dataset: MemeDatasetConfig = field(
        default_factory=lambda: MemeDatasetConfig(
            datasets=["Hateful_Memes_Dataset", "MAMI"], splits=["test"],
        )
    )
    identity_group_terms: list[str] = field(default_factory=lambda: list(IDENTITY_GROUP_TERMS))
    output_dir: str = "./identity_group_controls"


def run(cfg: BuildIdentityGroupControlsConfig) -> None:
    assert "train" not in cfg.dataset.splits, (
        "Refusing to build identity-group controls from a TRAIN split -- "
        "these controls audit SFT/RLFR models for false positives, and using "
        "data those models were trained on would make the audit meaningless. "
        "Use --dataset.splits test (or validation if no test split exists)."
    )
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    terms = tuple(cfg.identity_group_terms)

    samples = load_meme_samples(cfg.dataset)
    logger.info("Loaded %d samples across %s", len(samples), cfg.dataset.datasets)

    rows: list[dict[str, Any]] = []
    n_by_dataset: dict[str, int] = {}
    for sample in samples:
        gold = sample.gold_fields.get(sample.schema.primary_field.name)
        if gold is None:
            continue
        primary_labels = sample.schema.primary_field.labels
        positive_label = primary_labels[-1]
        negative_label = primary_labels[0]
        if gold != negative_label:
            continue  # controls must be GOLD-BENIGN by construction
        matched_term = contains_identity_group_term(sample.meme_text, terms)
        if matched_term is None:
            continue
        rows.append({
            "sample_id": sample.sample_id,
            "dataset_name": sample.dataset_name,
            "task_name": sample.task_name,
            "split": sample.split,
            "image_path": sample.image_path,
            "meme_text": sample.meme_text,
            "gold_label": gold,
            "positive_label": positive_label,
            "negative_label": negative_label,
            "matched_term": matched_term,
        })
        n_by_dataset[sample.dataset_name] = n_by_dataset.get(sample.dataset_name, 0) + 1

    assert rows, (
        "No benign identity-group control samples found -- check dataset/splits "
        "or extend reward_monitor.IDENTITY_GROUP_TERMS"
    )
    logger.info("Matched %d benign identity-group controls: %s", len(rows), n_by_dataset)

    with (out_dir / "identity_group_controls.jsonl").open("w") as fd:
        for row in rows:
            fd.write(json.dumps(row) + "\n")

    summary = {
        "n_controls": len(rows),
        "n_by_dataset": n_by_dataset,
        "identity_group_terms": list(terms),
    }
    (out_dir / "identity_group_controls_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %d controls to %s", len(rows), out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(BuildIdentityGroupControlsConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
