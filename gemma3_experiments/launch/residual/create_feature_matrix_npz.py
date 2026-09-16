

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.probe_validate import (
    _build_feature_blocks,
    _extract_features_multi,
    _read_jsonl,
    _required_token_selections,
    _row_gold_label,
)
from gemma3_experiments.residual_sae import (
    load_residual_sae,
    load_residual_stats,
    resolve_visible_device,
)

logger = logging.getLogger(__name__)


@dataclass
class CreateFeatureMatrixConfig:
    residual_sae_path: str = ""
    cache_dir: str = ""
    extra_cache_dirs: list[str] = field(default_factory=list)
    output_path: str = ""

    datasets: list[str] = field(default_factory=lambda: [
        "MMHS150K",
        "HarMeme",
        "Hateful_Memes_Dataset",
        "MAMI",
        "MultiOFF",
        "Crisis_HateMM",
    ])
    splits: list[str] = field(default_factory=lambda: ["validation", "val", "dev"])

    token_selection: str = "all"  # image | all | generated | image_and_generated | prompt
    concat_token_selections: list[str] = field(default_factory=list)
    interaction_token_selection_pairs: list[str] = field(default_factory=list)

    aggregation: str = "max"  # max | mean
    batch_size: int = 256
    device: str = "cuda:0"


def _load_rows(cache_dirs: list[Path], datasets: set[str], splits: set[str]) -> list[dict]:
    rows: list[dict] = []
    for cache_dir in cache_dirs:
        manifest_path = cache_dir / "manifest.jsonl"
        assert manifest_path.exists(), f"Manifest missing: {manifest_path}"
        source_rows = _read_jsonl(manifest_path)
        for row in source_rows:
            if row.get("dataset_name") not in datasets:
                continue
            if row.get("split") not in splits:
                continue
            if _row_gold_label(row) is None:
                continue
            tagged = dict(row)
            tagged["_cache_dir"] = str(cache_dir)
            rows.append(tagged)
        logger.info("Loaded %d rows from %s", len(source_rows), cache_dir)
    return rows


def _write_feature_matrix(path: Path, keys: list[str], X) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, keys=np.array(keys, dtype=object), X=X)


def run(cfg: CreateFeatureMatrixConfig) -> None:
    assert cfg.residual_sae_path, "residual_sae_path is required"
    assert cfg.cache_dir, "cache_dir is required"
    assert cfg.output_path, "output_path is required"

    cache_dirs = [Path(cfg.cache_dir)] + [Path(p) for p in cfg.extra_cache_dirs]
    for cache_dir in cache_dirs:
        assert cache_dir.exists(), f"Cache directory missing: {cache_dir}"

    device = resolve_visible_device(cfg.device)
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)
    n_features = sae.num_latents
    logger.info("Loaded SAE num_latents=%d", n_features)

    rows = _load_rows(cache_dirs, set(cfg.datasets), set(cfg.splits))
    assert rows, "No rows matched filters (datasets/splits/gold labels)."
    logger.info("Matched %d rows for extraction", len(rows))

    extract_kw = dict(
        sae=sae,
        stats=stats,
        aggregation=cfg.aggregation,
        batch_size=cfg.batch_size,
        device=device,
        n_features=n_features,
    )

    if cfg.concat_token_selections or cfg.interaction_token_selection_pairs:
        required = _required_token_selections(
            cfg.concat_token_selections,
            cfg.interaction_token_selection_pairs,
        )
        logger.info(
            "Extracting feature blocks for token selections=%s interactions=%s",
            cfg.concat_token_selections,
            cfg.interaction_token_selection_pairs,
        )
        feature_sets = {
            token_selection: _extract_features_multi(rows, token_selection=token_selection, **extract_kw)
            for token_selection in required
        }
        keys, X = _build_feature_blocks(
            feature_sets,
            cfg.concat_token_selections,
            cfg.interaction_token_selection_pairs,
        )
    else:
        logger.info("Extracting features for token_selection=%s", cfg.token_selection)
        keys, X = _extract_features_multi(rows, token_selection=cfg.token_selection, **extract_kw)

    output_path = Path(cfg.output_path)
    _write_feature_matrix(output_path, keys, X)
    logger.info("Wrote feature matrix: %s", output_path)
    logger.info("Shape=%s nnz=%d n_keys=%d", tuple(X.shape), int(X.nnz), len(keys))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(CreateFeatureMatrixConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
