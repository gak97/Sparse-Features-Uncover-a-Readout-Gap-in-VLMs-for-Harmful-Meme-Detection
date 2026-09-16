

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import Serializable, parse


logger = logging.getLogger(__name__)


@dataclass
class PrepareExistConfig(Serializable):
    dataset_root: str = "./data/EXIST_2025_Memes_Dataset"
    output_path: str = "./outputs/exist_multilingual/exist_split.jsonl"
    eval_per_language: int = 250
    seed: int = 20260730


def _strict_majority(labels: list[str]) -> str | None:
    normalized = [str(label).upper().strip() for label in labels]
    counts = Counter(normalized)
    if not counts or set(counts) - {"YES", "NO"}:
        return None
    top_label, top_count = counts.most_common(1)[0]
    if list(counts.values()).count(top_count) != 1:
        return None
    return top_label


def _stratified_eval_ids(rows: list[dict], n_eval: int, rng: random.Random) -> set[str]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row["sample_id"])
    if len(by_label) != 2:
        raise ValueError(f"Expected both YES and NO labels, found {sorted(by_label)}")

    for sample_ids in by_label.values():
        rng.shuffle(sample_ids)
    n_yes = round(n_eval * len(by_label["YES"]) / len(rows))
    n_yes = max(1, min(n_yes, len(by_label["YES"]) - 1))
    n_no = n_eval - n_yes
    if n_no < 1 or n_no >= len(by_label["NO"]):
        raise ValueError(f"Cannot draw {n_eval} stratified examples from class counts {dict(map(lambda item: (item[0], len(item[1])), by_label.items()))}")
    return set(by_label["YES"][:n_yes] + by_label["NO"][:n_no])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = parse(PrepareExistConfig)
    root = Path(cfg.dataset_root)
    source_path = root / "training" / "EXIST2025_training.json"
    image_root = source_path.parent
    raw_records = json.loads(source_path.read_text())
    if not isinstance(raw_records, dict):
        raise ValueError(f"Expected a JSON object in {source_path}, got {type(raw_records).__name__}")

    rows: list[dict] = []
    excluded = Counter()
    for record_id, raw in raw_records.items():
        language = str(raw.get("lang", "")).lower()
        if language not in {"en", "es"}:
            excluded["unknown_language"] += 1
            continue
        label = _strict_majority(raw.get("labels_task2_1", []))
        if label is None:
            excluded["ambiguous_or_invalid_vote"] += 1
            continue
        image_path = image_root / str(raw.get("path_memes", ""))
        if not image_path.is_file():
            excluded["missing_image"] += 1
            continue
        rows.append({
            "dataset_name": "EXIST2025_Memes",
            "task_name": "sexism_detection",
            "sample_id": str(raw.get("id_EXIST", record_id)),
            "language": language,
            "label": label,
            "gold_fields": {"label": "sexist" if label == "YES" else "non-sexist"},
            "image_path": str(image_path),
            "meme_text": str(raw.get("text", "")).strip(),
            "raw_labels_task2_1": raw.get("labels_task2_1", []),
            "source_split": raw.get("split"),
        })

    by_language: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)
    if set(by_language) != {"en", "es"}:
        raise ValueError(f"Expected English and Spanish records, found {sorted(by_language)}")

    eval_ids: set[str] = set()
    for language in ("en", "es"):
        local_rng = random.Random(f"{cfg.seed}:{language}")
        language_eval_ids = _stratified_eval_ids(by_language[language], cfg.eval_per_language, local_rng)
        if eval_ids & language_eval_ids:
            raise AssertionError("Language partitions unexpectedly overlap")
        eval_ids.update(language_eval_ids)

    for row in rows:
        row["split"] = "eval" if row["sample_id"] in eval_ids else "probe_train"

    train_ids = {row["sample_id"] for row in rows if row["split"] == "probe_train"}
    if train_ids & eval_ids:
        raise AssertionError("Training and evaluation IDs overlap")

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fd:
        for row in sorted(rows, key=lambda item: (item["language"], item["sample_id"])):
            fd.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source": str(source_path),
        "seed": cfg.seed,
        "eval_per_language": cfg.eval_per_language,
        "n_total": len(rows),
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "counts": {
            split: {
                language: dict(Counter(row["label"] for row in rows if row["split"] == split and row["language"] == language))
                for language in ("en", "es")
            }
            for split in ("probe_train", "eval")
        },
        "excluded": dict(excluded),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote %d rows (%d train, %d eval) to %s", len(rows), len(train_ids), len(eval_ids), output_path)
    logger.info("Summary: %s", json.dumps(summary["counts"], sort_keys=True))


if __name__ == "__main__":
    main()