"""Download small, frozen COCO-caption and VQAv2 capability sanity sets."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)


@dataclass
class DownloadReleaseCapabilitySetsConfig:
    output_dir: str = "./data/gemma_release_capability_sets"
    hf_cache_dir: str = "./data/huggingface"
    caption_dataset: str = "Multimodal-Fatima/COCO_captions_validation"
    caption_split: str = "validation"
    vqa_dataset: str = "Multimodal-Fatima/VQAv2_sample_validation"
    vqa_split: str = "validation"
    n_caption_examples: int = 128
    n_vqa_examples: int = 128
    seed: int = 20260818


def _first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("text", "caption", "captions", "answer", "answers"):
            if key in value:
                return _strings(value[key])
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_strings(item))
        return result
    return []


def _save_examples(dataset: Any, n_examples: int, out_dir: Path, kind: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(dataset.shuffle(seed=seed, buffer_size=512).take(n_examples)):
        image = _first(row, ("image", "img"))
        assert image is not None, f"{kind} row {i} has no image; available fields={list(row)}"
        image_path = images_dir / f"{kind}_{i:04d}.png"
        image.convert("RGB").save(image_path)
        if kind == "caption":
            answers = _strings(_first(row, ("sentences_raw", "sentences", "caption", "captions", "text")))
            prompt = "Describe this image accurately in one sentence."
        else:
            answers = _strings(_first(row, ("answers", "answer", "multiple_choice_answer")))
            prompt = f"Answer this question about the image briefly: {_first(row, ('question', 'query'))}"
        assert answers, f"{kind} row {i} has no references; available fields={list(row)}"
        rows.append({"kind": kind, "sample_id": f"{kind}_{i:04d}", "image_path": str(image_path), "prompt": prompt, "reference_answers": answers})
    return rows


def run(cfg: DownloadReleaseCapabilitySetsConfig) -> None:
    from datasets import load_dataset

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caption = load_dataset(cfg.caption_dataset, split=cfg.caption_split, cache_dir=cfg.hf_cache_dir, streaming=True)
    vqa = load_dataset(cfg.vqa_dataset, split=cfg.vqa_split, cache_dir=cfg.hf_cache_dir, streaming=True)
    rows = _save_examples(caption, cfg.n_caption_examples, out_dir, "caption", cfg.seed)
    rows.extend(_save_examples(vqa, cfg.n_vqa_examples, out_dir, "vqa", cfg.seed + 1))
    with (out_dir / "capability_manifest.jsonl").open("w") as fd:
        for row in rows:
            fd.write(json.dumps(row) + "\n")
    summary = {"n_caption": cfg.n_caption_examples, "n_vqa": cfg.n_vqa_examples, "caption_dataset": cfg.caption_dataset, "vqa_dataset": cfg.vqa_dataset, "seed": cfg.seed}
    (out_dir / "capability_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote frozen capability manifests to %s", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse(DownloadReleaseCapabilitySetsConfig, argument_generation_mode=ArgumentGenerationMode.BOTH))


if __name__ == "__main__":
    main()