

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.probe_validate import (
    _aggregate_sample,
    _read_jsonl,
    _row_gold_label,
    _row_sample_key,
    _row_task_key,
)
from gemma3_experiments.residual_sae import load_residual_sae, load_residual_stats, resolve_visible_device

logger = logging.getLogger(__name__)


@dataclass
class ResidualFeatureExemplarConfig:
    residual_sae_path: str = ""
    residual_cache_dir: str = ""
    output_dir: str = ""
    top_features_paths: list[str] = field(default_factory=list)
    extra_feature_indices: list[int] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=lambda: ["train", "validation", "test"])
    token_selection: str = "generated"
    aggregation: str = "max"
    batch_size: int = 256
    top_k_examples: int = 25
    max_features: int = 300
    device: str = "cuda:0"


def _collect_feature_indices(cfg: ResidualFeatureExemplarConfig) -> list[int]:
    feature_i = set(int(i) for i in cfg.extra_feature_indices)
    for path_str in cfg.top_features_paths:
        path = Path(path_str)
        if not path.exists():
            logger.warning("Top-features path missing: %s", path)
            continue
        if path.suffix == ".jsonl":
            for row in _read_jsonl(path):
                if "feature_idx" in row:
                    feature_i.add(int(row["feature_idx"]))
                for feat in row.get("top_features", []) or []:
                    if "feature_idx" in feat:
                        feature_i.add(int(feat["feature_idx"]))
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            for by_class in data.values():
                if not isinstance(by_class, dict):
                    continue
                for feats in by_class.values():
                    if not isinstance(feats, list):
                        continue
                    for feat in feats:
                        if isinstance(feat, dict) and "feature_idx" in feat:
                            feature_i.add(int(feat["feature_idx"]))
    out = sorted(feature_i)
    if cfg.max_features and len(out) > cfg.max_features:
        logger.info("Truncating selected features from %d to %d", len(out), cfg.max_features)
        out = out[: cfg.max_features]
    return out


def _push_top(top: dict[int, list[dict[str, Any]]], feature_i: int, row: dict[str, Any], activation: float, top_k: int) -> None:
    item = {
        "activation": round(float(activation), 6),
        "sample_key": _row_sample_key(row),
        "task_key": _row_task_key(row),
        "dataset_name": row.get("dataset_name"),
        "task_name": row.get("task_name"),
        "split": row.get("split"),
        "gold_label": _row_gold_label(row),
        "meme_text": row.get("meme_text"),
        "generated_text": row.get("generated_text"),
        "image_path": row.get("image_path"),
    }
    bucket = top[feature_i]
    bucket.append(item)
    bucket.sort(key=lambda x: x["activation"], reverse=True)
    if len(bucket) > top_k:
        del bucket[top_k:]


def run(cfg: ResidualFeatureExemplarConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.residual_cache_dir)
    assert cache_dir.exists(), f"Residual cache missing: {cache_dir}"
    feature_i = _collect_feature_indices(cfg)
    assert feature_i, "No feature indices selected."
    feature_tensor_i = torch.tensor(feature_i, dtype=torch.long)
    feature_pos_by_idx = {idx: pos for pos, idx in enumerate(feature_i)}
    logger.info("Selected %d features", len(feature_i))

    device = resolve_visible_device(cfg.device)
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)
    feature_tensor_i_device = feature_tensor_i.to(device)

    rows = _read_jsonl(cache_dir / "manifest.jsonl")
    dataset_filter = set(cfg.datasets)
    split_filter = set(cfg.splits)
    rows = [
        row for row in rows
        if (not dataset_filter or row.get("dataset_name") in dataset_filter)
        and row.get("split") in split_filter
        and _row_gold_label(row) is not None
    ]
    logger.info("Rows selected for exemplar extraction: %d", len(rows))

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)

    top: dict[int, list[dict[str, Any]]] = {idx: [] for idx in feature_i}
    for shard_i, (shard_file, shard_rows) in enumerate(sorted(by_shard.items()), start=1):
        shard = load_file(str(cache_dir / shard_file))
        residual_all = shard["residual"]
        roles_all = shard["token_roles"]
        for row in sorted(shard_rows, key=lambda r: int(r["offset"])):
            offset = int(row["offset"])
            length = int(row["length"])
            vec = _aggregate_sample(
                sae,
                residual_all[offset: offset + length],
                roles_all[offset: offset + length],
                stats=stats,
                token_selection=cfg.token_selection,
                batch_size=cfg.batch_size,
                device=device,
                aggregation=cfg.aggregation,
            )
            if vec is None:
                continue
            selected = vec.to(device)[feature_tensor_i_device].detach().cpu()
            for pos, value in enumerate(selected.tolist()):
                if value <= 0:
                    continue
                _push_top(top, feature_i[pos], row, value, cfg.top_k_examples)
        logger.info("Processed shard %d/%d (%s)", shard_i, len(by_shard), shard_file)

    payload = {
        "config": {**vars(cfg), "selected_feature_count": len(feature_i)},
        "features": {str(idx): examples for idx, examples in top.items()},
    }
    (output_dir / "feature_exemplars.json").write_text(json.dumps(payload, indent=2))
    with (output_dir / "feature_exemplars.jsonl").open("w") as fd:
        for idx, examples in top.items():
            fd.write(json.dumps({"feature_idx": idx, "examples": examples}) + "\n")
    logger.info("Wrote feature exemplars to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ResidualFeatureExemplarConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
