

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from simple_parsing import Serializable, list_field, parse

from gemma3_experiments.residual_sae.datasets import MemeSample, TaskField, TaskSchema

from ..config import DenseActivationCacheConfig, QwenResidualConfig
from .cache_dense import run_dense_activation_cache


logger = logging.getLogger(__name__)

EXIST_SCHEMA = TaskSchema(
    task_name="sexism_detection",
    task_description="Decide whether the meme contains sexist or misogynistic content.",
    primary_field=TaskField("label", ("non-sexist", "sexist")),
)


@dataclass
class ExistDenseCacheConfig(Serializable):
    manifest_path: str = "./outputs/exist_multilingual/exist_split.jsonl"
    output_dir: str = "./outputs/exist_multilingual/dense_raw"
    prompt_variant: str = "raw_ocr"  # raw_ocr | no_ocr
    splits: list[str] = list_field("probe_train", "eval")
    languages: list[str] = list_field("en", "es")
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    model_device: str = "cuda:0"
    hook_module_path: str = "model.language_model.layers.20"
    token_selection: str = "all"
    capture_generation: bool = True
    max_new_tokens: int = 96
    shard_size: int = 32
    overwrite: bool = False
    resume: bool = False
    dry_run: bool = False


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fd:
        return [json.loads(line) for line in fd if line.strip()]


def load_exist_samples(cfg: ExistDenseCacheConfig) -> list[MemeSample]:
    wanted_splits, wanted_languages = set(cfg.splits), set(cfg.languages)
    samples: list[MemeSample] = []
    for row in _read_jsonl(Path(cfg.manifest_path)):
        language = row["language"]
        if row["split"] not in wanted_splits or language not in wanted_languages:
            continue
        if cfg.prompt_variant == "raw_ocr":
            meme_text = row["meme_text"]
        elif cfg.prompt_variant == "no_ocr":
            meme_text = ""
        else:
            raise ValueError(f"Unsupported prompt_variant: {cfg.prompt_variant!r}")

        samples.append(MemeSample(
            dataset_name=row["dataset_name"],
            task_name=row["task_name"],
            sample_id=row["sample_id"],
            split=row["split"],
            image_path=row["image_path"],
            meme_text=meme_text,
            schema=EXIST_SCHEMA,
            gold_fields=row["gold_fields"],
            raw_fields={
                "language": language,
                "majority_label": row["label"],
                "prompt_variant": cfg.prompt_variant,
                "ocr_text": row["meme_text"],
                "source_split": row.get("source_split"),
            },
        ))
    return samples


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ExistDenseCacheConfig)
    samples = load_exist_samples(cfg)
    by_language: dict[str, int] = {}
    for sample in samples:
        language = str(sample.raw_fields["language"])
        by_language[language] = by_language.get(language, 0) + 1
    logger.info("EXIST cache input: n=%d languages=%s variant=%s", len(samples), by_language, cfg.prompt_variant)
    if not samples:
        raise ValueError("No EXIST samples matched the requested splits/languages")
    if cfg.dry_run:
        return

    cache_cfg = DenseActivationCacheConfig(
        output_dir=cfg.output_dir,
        shard_size=cfg.shard_size,
        overwrite=cfg.overwrite,
        resume=cfg.resume,
        qwen=QwenResidualConfig(
            model_id=cfg.model_id,
            model_device=cfg.model_device,
            hook_module_path=cfg.hook_module_path,
            token_selection=cfg.token_selection,
            capture_generation=cfg.capture_generation,
            max_new_tokens=cfg.max_new_tokens,
        ),
    )
    run_dense_activation_cache(cache_cfg, samples=samples)


if __name__ == "__main__":
    main()