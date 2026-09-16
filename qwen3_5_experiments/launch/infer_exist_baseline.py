"""Run Qwen's constrained yes/no baseline on the locked EXIST holdout."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from simple_parsing import Serializable, parse

from ..config import QwenResidualConfig
from ..qwen import classify_constrained, load_qwen3_model
from .analyze_exist_multilingual import _metrics, _read_jsonl

logger = logging.getLogger(__name__)


@dataclass
class ExistBaselineConfig(Serializable):
    cache_dir: str = "./outputs/exist_multilingual/dense_raw"
    output_dir: str = "./outputs/exist_multilingual/baseline_raw_ocr"
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    device: str = "cuda:0"
    resume: bool = True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ExistBaselineConfig)
    cache_dir, output_dir = Path(cfg.cache_dir), Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline_eval.jsonl"
    completed = {json.loads(line)["sample_id"] for line in output_path.open()} if cfg.resume and output_path.exists() else set()
    rows = [
        row for row in _read_jsonl(cache_dir / "manifest.jsonl")
        if row.get("split") == "eval" and row.get("task_name") == "sexism_detection" and row["sample_id"] not in completed
    ]
    qwen_cfg = QwenResidualConfig(model_id=cfg.model_id, model_device=cfg.device)
    model, processor = load_qwen3_model(qwen_cfg)
    with output_path.open("a") as fd:
        for index, row in enumerate(rows, start=1):
            image = Image.open(row["image_path"]).convert("RGB")
            predicted, text, confidence = classify_constrained(model, processor, image, row["prompt_text"], cfg.device)
            fd.write(json.dumps({
                "sample_id": row["sample_id"],
                "language": row.get("raw_fields", {}).get("language"),
                "gold": row["gold_fields"]["label"],
                "prediction": "sexist" if predicted else "non-sexist",
                "prediction_text": text,
                "conf_gap": confidence,
            }) + "\n")
            fd.flush()
            if index % 25 == 0 or index == len(rows):
                logger.info("Baseline progress: %d/%d", index, len(rows))

    completed_rows = _read_jsonl(output_path)
    summary: dict[str, dict] = {}
    for language in ("en", "es", "all"):
        subset = completed_rows if language == "all" else [row for row in completed_rows if row["language"] == language]
        gold = np.asarray([int(row["gold"] == "sexist") for row in subset], dtype=np.int32)
        pred = np.asarray([int(row["prediction"] == "sexist") for row in subset], dtype=np.int32)
        summary[language] = {"n_eval": len(subset), **_metrics(gold, pred)}
    (output_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote baseline outputs and summary to %s", output_dir)


if __name__ == "__main__":
    main()
