

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from simple_parsing import Serializable, parse


logger = logging.getLogger(__name__)

LABEL_MAP = {"Bully": "bully", "Nonbully": "non-bully"}


@dataclass
class PrepareMultiBullyConfig(Serializable):
    dataset_root: str = "./data/MultiBully"
    output_path: str = "./outputs/multibully/multibully_split.jsonl"
    eval_size: int = 1000
    seed: int = 20260803


def _is_decodable_image(path: Path) -> bool:
    """Fully load an image so cache workers never encounter a corrupt file."""
    try:
        with Image.open(path) as image:
            image.convert("RGB").load()
    except (UnidentifiedImageError, OSError, ValueError):
        return False
    return True


def _stratified_eval_ids(rows: list[dict], eval_size: int, rng: random.Random) -> set[str]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row["sample_id"])
    if set(by_label) != set(LABEL_MAP):
        raise ValueError(f"Expected labels {sorted(LABEL_MAP)}, found {sorted(by_label)}")

    for ids in by_label.values():
        rng.shuffle(ids)
    n_bully = round(eval_size * len(by_label["Bully"]) / len(rows))
    n_bully = max(1, min(n_bully, len(by_label["Bully"]) - 1))
    n_nonbully = eval_size - n_bully
    if n_nonbully < 1 or n_nonbully >= len(by_label["Nonbully"]):
        raise ValueError(f"Cannot draw {eval_size} examples from class counts { {label: len(ids) for label, ids in by_label.items()} }")
    return set(by_label["Bully"][:n_bully] + by_label["Nonbully"][:n_nonbully])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse(PrepareMultiBullyConfig)
    root = Path(cfg.dataset_root)
    workbook = root / "Cyberbully_corrected_emotion_sentiment.xlsx"
    image_root = root / "bully_data"
    if not workbook.is_file() or not image_root.is_dir():
        raise FileNotFoundError(f"Expected workbook={workbook} and images={image_root}")

    dataframe = pd.read_excel(workbook, sheet_name="cyberbully")
    required = {"Img-Name", "Img-Text", "Img-Text-Label", "Img-Label", "Text-Label"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Workbook missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    excluded = Counter()
    seen_ids: set[str] = set()
    for _, record in dataframe.iterrows():
        image_name = str(record["Img-Name"]).strip() if pd.notna(record["Img-Name"]) else ""
        joint_label = str(record["Img-Text-Label"]).strip() if pd.notna(record["Img-Text-Label"]) else ""
        if joint_label not in LABEL_MAP:
            excluded["missing_or_invalid_joint_label"] += 1
            continue
        if not image_name:
            excluded["missing_image_name"] += 1
            continue
        image_path = image_root / image_name
        if not image_path.is_file():
            excluded["missing_image_file"] += 1
            continue
        if not _is_decodable_image(image_path):
            excluded["unreadable_image_file"] += 1
            continue
        if image_name in seen_ids:
            excluded["duplicate_image_name"] += 1
            continue
        seen_ids.add(image_name)
        meme_text = str(record["Img-Text"]).strip() if pd.notna(record["Img-Text"]) else ""
        rows.append({
            "dataset_name": "MultiBully",
            "task_name": "cyberbullying_detection",
            "sample_id": image_name,
            "language": "hi_en_code_mixed",
            "label": joint_label,
            "gold_fields": {"label": LABEL_MAP[joint_label]},
            "image_path": str(image_path),
            "meme_text": meme_text,
            "raw_labels": {
                "joint": joint_label,
                "image": str(record["Img-Label"]).strip() if pd.notna(record["Img-Label"]) else "",
                "text": str(record["Text-Label"]).strip() if pd.notna(record["Text-Label"]) else "",
                "harmful_score": str(record.get("Harmful-Score", "")).strip() if pd.notna(record.get("Harmful-Score", "")) else "",
                "target": str(record.get("Target", "")).strip() if pd.notna(record.get("Target", "")) else "",
            },
        })

    if cfg.eval_size >= len(rows):
        raise ValueError(f"eval_size={cfg.eval_size} must be smaller than usable rows={len(rows)}")
    eval_ids = _stratified_eval_ids(rows, cfg.eval_size, random.Random(cfg.seed))
    for row in rows:
        row["split"] = "eval" if row["sample_id"] in eval_ids else "probe_train"

    train_ids = {row["sample_id"] for row in rows if row["split"] == "probe_train"}
    if train_ids & eval_ids or len(eval_ids) != cfg.eval_size:
        raise AssertionError("Invalid train/evaluation partition")

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fd:
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            fd.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source": str(workbook),
        "label_source": "Img-Text-Label",
        "seed": cfg.seed,
        "eval_size": cfg.eval_size,
        "n_total": len(rows),
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "counts": {
            split: dict(Counter(row["label"] for row in rows if row["split"] == split))
            for split in ("probe_train", "eval")
        },
        "excluded": dict(excluded),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote %d rows (%d train, %d eval) to %s", len(rows), len(train_ids), len(eval_ids), output_path)
    logger.info("Summary: %s", json.dumps(summary["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
