"""Cache Qwen layer-20 states for the leakage-free MultiBully manifest."""

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

MULTIBULLY_SCHEMA = TaskSchema(
    task_name="cyberbullying_detection",
    task_description="Decide whether the meme contains cyberbullying, harassment, or targeted abusive content.",
    primary_field=TaskField("label", ("non-bully", "bully")),
)


@dataclass
class MultiBullyDenseCacheConfig(Serializable):
    manifest_path: str = "./outputs/multibully/multibully_split.jsonl"
    output_dir: str = "./outputs/multibully/dense_raw"
    prompt_variant: str = "raw_ocr"  # raw_ocr | no_ocr
    splits: list[str] = list_field("probe_train", "eval")
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


def load_multibully_samples(cfg: MultiBullyDenseCacheConfig) -> list[MemeSample]:
    samples: list[MemeSample] = []
    for row in _read_jsonl(Path(cfg.manifest_path)):
        if row["split"] not in set(cfg.splits):
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
            schema=MULTIBULLY_SCHEMA,
            gold_fields=row["gold_fields"],
            raw_fields={
                "language": row["language"],
                "joint_label": row["label"],
                "prompt_variant": cfg.prompt_variant,
                "ocr_text": row["meme_text"],
                **row["raw_labels"],
            },
        ))
    return samples


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(MultiBullyDenseCacheConfig)
    samples = load_multibully_samples(cfg)
    logger.info("MultiBully cache input: n=%d variant=%s", len(samples), cfg.prompt_variant)
    if not samples:
        raise ValueError("No MultiBully samples matched the requested splits")
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
