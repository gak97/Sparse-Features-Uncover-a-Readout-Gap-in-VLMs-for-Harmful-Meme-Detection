

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from simple_parsing import parse, Serializable, list_field

from ..qwen import QwenSae

logger = logging.getLogger(__name__)

# Token role constants (match cache_dense.py / discover_features.py)
ROLE_PROMPT = 0
ROLE_IMAGE = 1
ROLE_GENERATED = 2


@dataclass
class BaseSaeFeatureCardConfig(Serializable):
    dense_cache_dirs: list[str] = list_field()
    top_features_path: str = "./qwen_probe_base_sae/top_features.json"
    extra_feature_indices: list[int] = list_field()

    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    device: str = "cuda:0"

    splits: list[str] = list_field("train", "validation", "test", "test_seen")
    token_selection: str = "all"   # all | image | prompt | generated
    aggregation: str = "max"       # max | mean
    max_features_per_direction: int = 20   # how many of label_up/label_down to build cards for
    top_k_examples: int = 15
    output_dir: str = "./qwen_base_sae_feature_cards"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fd:
        for line in fd:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _role_mask(roles: torch.Tensor, selection: str) -> torch.Tensor:
    if selection == "all":
        return torch.ones_like(roles, dtype=torch.bool)
    if selection == "image":
        return roles == ROLE_IMAGE
    if selection == "prompt":
        return roles == ROLE_PROMPT
    if selection == "generated":
        return roles == ROLE_GENERATED
    raise ValueError(f"Unknown token_selection: {selection!r}")


def _collect_feature_indices(cfg: BaseSaeFeatureCardConfig) -> tuple[set[int], dict]:
    """Returns (all_feature_indices, top_features_data) truncated per direction."""
    path = Path(cfg.top_features_path)
    data: dict[str, dict] = {}
    if path.exists():
        data = json.loads(path.read_text())

    feature_i: set[int] = set(int(i) for i in cfg.extra_feature_indices)
    truncated: dict[str, dict] = {}
    for task_key, by_dir in data.items():
        truncated[task_key] = {}
        for direction in ("label_up", "label_down"):
            feats = by_dir.get(direction, [])[: cfg.max_features_per_direction]
            truncated[task_key][direction] = feats
            for f in feats:
                feature_i.add(int(f["feature_idx"]))
    return feature_i, truncated


def _push_top(top: dict[int, list[dict[str, Any]]], feature_i: int, row: dict[str, Any], activation: float, top_k: int) -> None:
    item = {
        "activation": round(float(activation), 6),
        "dataset_name": row.get("dataset_name"),
        "task_name": row.get("task_name"),
        "sample_id": row.get("sample_id"),
        "split": row.get("split"),
        "gold_fields": row.get("gold_fields"),
        "meme_text": row.get("meme_text"),
        "generated_text": row.get("generated_text"),
        "image_path": row.get("image_path"),
    }
    bucket = top[feature_i]
    bucket.append(item)
    bucket.sort(key=lambda x: x["activation"], reverse=True)
    if len(bucket) > top_k:
        del bucket[top_k:]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(BaseSaeFeatureCardConfig)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_i, top_features_truncated = _collect_feature_indices(cfg)
    assert feature_i, f"No feature indices found (check --top_features_path={cfg.top_features_path})"
    feature_list = sorted(feature_i)
    logger.info("Selected %d unique base-SAE features across all tasks/directions", len(feature_list))

    base_sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    logger.info("Base SAE loaded: d_model=%d d_sae=%d k=%d", base_sae.cfg.d_model, base_sae.cfg.d_sae, base_sae.cfg.k)

    feature_tensor = torch.tensor(feature_list, dtype=torch.long, device=cfg.device)
    top: dict[int, list[dict[str, Any]]] = {idx: [] for idx in feature_list}

    split_filter = set(cfg.splits)
    n_rows_seen = 0
    for cache_dir_str in cfg.dense_cache_dirs:
        cache_dir = Path(cache_dir_str)
        manifest_path = cache_dir / "manifest.jsonl"
        if not manifest_path.exists():
            logger.warning("Missing manifest: %s", manifest_path)
            continue
        rows = [r for r in _read_jsonl(manifest_path) if r.get("split") in split_filter]
        logger.info("%s: %d rows in splits %s", cache_dir, len(rows), cfg.splits)

        by_shard: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_shard[row["shard_file"]].append(row)

        for shard_i, (shard_name, shard_rows) in enumerate(sorted(by_shard.items()), start=1):
            shard_path = cache_dir / shard_name
            if not shard_path.exists():
                logger.warning("Missing shard: %s", shard_path)
                continue
            try:
                tensors = load_file(str(shard_path))
            except Exception as e:
                logger.warning("Failed to load %s: %s", shard_path, e)
                continue

            hidden_all = tensors["hidden"].to(device=cfg.device, dtype=torch.float32)
            roles_all = tensors["token_roles"].to(device=cfg.device)

            for row in shard_rows:
                offset, length = row["offset"], row["length"]
                sample_h = hidden_all[offset: offset + length]
                sample_roles = roles_all[offset: offset + length]
                mask = _role_mask(sample_roles, cfg.token_selection)
                sample_h = sample_h[mask]
                if sample_h.shape[0] == 0:
                    continue

                with torch.no_grad():
                    acts, _ = base_sae.encode(sample_h)  # [T', d_sae]
                    if cfg.aggregation == "max":
                        vec = acts.max(dim=0).values
                    else:
                        vec = acts.mean(dim=0)

                selected = vec[feature_tensor].detach().cpu()
                for pos, value in enumerate(selected.tolist()):
                    if value <= 0:
                        continue
                    _push_top(top, feature_list[pos], row, value, cfg.top_k_examples)
                n_rows_seen += 1

            if shard_i % 50 == 0 or shard_i == len(by_shard):
                logger.info("  %s: processed shard %d/%d (rows seen so far: %d)", cache_dir, shard_i, len(by_shard), n_rows_seen)

    # Assemble feature cards keyed by task/direction (mirrors top_features.json structure)
    cards: dict[str, dict] = {}
    for task_key, by_dir in top_features_truncated.items():
        cards[task_key] = {}
        for direction, feats in by_dir.items():
            cards[task_key][direction] = [
                {
                    "feature_idx": f["feature_idx"],
                    "weight": f["weight"],
                    "top_examples": top.get(int(f["feature_idx"]), []),
                }
                for f in feats
            ]

    out_path = output_dir / "feature_cards.json"
    out_path.write_text(json.dumps(cards, indent=2))
    logger.info("Wrote feature cards for %d tasks to %s (rows scanned: %d)", len(cards), out_path, n_rows_seen)

    # Quick human-readable summary
    for task_key, by_dir in cards.items():
        for direction, feats in by_dir.items():
            if not feats:
                continue
            top_feat = feats[0]
            n_ex = len(top_feat["top_examples"])
            logger.info(
                "%s [%s] top feature=%d weight=%.4f n_examples=%d",
                task_key, direction, top_feat["feature_idx"], top_feat["weight"], n_ex,
            )


if __name__ == "__main__":
    main()
