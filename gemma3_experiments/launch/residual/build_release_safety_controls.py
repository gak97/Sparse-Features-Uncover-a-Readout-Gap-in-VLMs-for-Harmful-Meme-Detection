

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.residual_sae import MemeDatasetConfig, load_meme_samples

logger = logging.getLogger(__name__)

IDENTITY_TERMS = (
    "black", "white", "asian", "jew", "muslim", "christian", "gay", "lesbian",
    "trans", "woman", "women", "female", "girl", "girls", "man", "men",
)
SLUR_TERMS = ("nigger", "nigga", "faggot", "dyke", "kike", "spic", "tranny")
RECLAIM_OR_QUOTE_TERMS = ("my nigga", "we niggas", "quoted", "quote", "said", "lyrics", "song")
DIRECT_HARM_TERMS = (
    "kill", "murder", "shoot", "hang", "gas", "exterminate", "death to", "should die",
    "nigger", "faggot", "kike", "spic", "tranny",
)
WOMEN_TERMS = ("woman", "women", "female", "girl", "girls", "wife", "mother", "mom")


@dataclass
class BuildReleaseSafetyControlsConfig:
    dataset: MemeDatasetConfig = field(
        default_factory=lambda: MemeDatasetConfig(
            datasets=["Hateful_Memes_Dataset", "MAMI", "HarMeme"], splits=["test"],
        )
    )
    max_per_control: int = 250
    output_dir: str = "./release_safety_controls"


def _contains(text: str, terms: tuple[str, ...]) -> list[str]:
    lower = text.lower()
    return [term for term in terms if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lower)]


def _is_harmful(sample: Any, gold: str) -> bool:
    labels = sample.schema.primary_field.labels
    if len(labels) == 2:
        return gold == labels[-1]
    return gold != labels[0]


def _control_row(sample: Any, gold: str, control: str, matched_terms: list[str]) -> dict[str, Any]:
    return {
        "control": control,
        "sample_id": sample.sample_id,
        "dataset_name": sample.dataset_name,
        "task_name": sample.task_name,
        "split": sample.split,
        "image_path": sample.image_path,
        "meme_text": sample.meme_text,
        "gold_label": gold,
        "is_harmful": _is_harmful(sample, gold),
        "matched_terms": matched_terms,
    }


def run(cfg: BuildReleaseSafetyControlsConfig) -> None:
    assert "train" not in cfg.dataset.splits, "Release controls must be held out; use test or validation only."
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    controls: dict[str, list[dict[str, Any]]] = {
        "benign_identity_group": [],
        "reclaimed_or_quoted_slur": [],
        "neutral_women_centered": [],
        "direct_threat_or_slur": [],
        "out_of_domain_harmful": [],
    }
    seen: dict[str, set[tuple[str, str, str]]] = {name: set() for name in controls}

    for sample in load_meme_samples(cfg.dataset):
        gold = sample.gold_fields.get(sample.schema.primary_field.name)
        if gold is None:
            continue
        text = sample.meme_text or ""
        is_harmful = _is_harmful(sample, gold)
        candidates = {
            "benign_identity_group": not is_harmful and _contains(text, IDENTITY_TERMS),
            "reclaimed_or_quoted_slur": not is_harmful and _contains(text, SLUR_TERMS) and _contains(text, RECLAIM_OR_QUOTE_TERMS),
            "neutral_women_centered": not is_harmful and _contains(text, WOMEN_TERMS),
            "direct_threat_or_slur": is_harmful and _contains(text, DIRECT_HARM_TERMS),
            "out_of_domain_harmful": sample.dataset_name == "HarMeme" and is_harmful,
        }
        for control, terms in candidates.items():
            key = (sample.dataset_name, sample.task_name, sample.sample_id)
            if not terms or key in seen[control] or len(controls[control]) >= cfg.max_per_control:
                continue
            seen[control].add(key)
            controls[control].append(_control_row(sample, gold, control, terms))

    empty = [name for name, rows in controls.items() if not rows]
    if empty:
        logger.warning("No held-out examples matched control slices: %s", empty)
    with (out_dir / "release_safety_controls.jsonl").open("w") as fd:
        for rows in controls.values():
            for row in rows:
                fd.write(json.dumps(row) + "\n")
    summary = {
        "dataset": {"datasets": cfg.dataset.datasets, "splits": cfg.dataset.splits},
        "n_by_control": {name: len(rows) for name, rows in controls.items()},
        "unavailable_controls": empty,
        "max_per_control": cfg.max_per_control,
    }
    (out_dir / "release_safety_controls_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote release controls: %s", summary["n_by_control"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse(BuildReleaseSafetyControlsConfig, argument_generation_mode=ArgumentGenerationMode.BOTH))


if __name__ == "__main__":
    main()