import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.residual_sae import (
    ROLE_GENERATED,
    ROLE_IMAGE,
    ROLE_PROMPT,
    ResidualFeatureDiscoveryConfig,
    load_residual_sae,
    load_residual_stats,
    resolve_visible_device,
)

logger = logging.getLogger(__name__)


@dataclass
class _GroupStats:
    n: int
    act_sum: torch.Tensor
    active_sum: torch.Tensor


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _role_mask(roles: torch.Tensor, token_selection: str) -> torch.Tensor:
    if token_selection == "all":
        return torch.ones_like(roles, dtype=torch.bool)
    if token_selection == "image":
        return roles == ROLE_IMAGE
    if token_selection == "generated":
        return roles == ROLE_GENERATED
    if token_selection == "image_and_generated":
        return (roles == ROLE_IMAGE) | (roles == ROLE_GENERATED)
    if token_selection == "prompt":
        return roles == ROLE_PROMPT
    raise ValueError(f"Unsupported token_selection: {token_selection!r}")


def _row_task_key(row: dict[str, Any]) -> str:
    primary = row["schema"]["primary_field"]
    return f"{row['dataset_name']}::{row['task_name']}::{primary['name']}"


def _row_gold_label(row: dict[str, Any]) -> str | None:
    primary = row["schema"]["primary_field"]
    value = row.get("gold_fields", {}).get(primary["name"])
    return value if isinstance(value, str) else None


def _filter_manifest_rows(rows: list[dict[str, Any]], cfg: ResidualFeatureDiscoveryConfig) -> list[dict[str, Any]]:
    datasets = set(cfg.dataset.datasets)
    splits = set(cfg.dataset.splits)
    out = [
        row
        for row in rows
        if row.get("dataset_name") in datasets
        and row.get("split") in splits
        and (_row_gold_label(row) is not None or cfg.dataset.include_unlabeled)
    ]
    if cfg.max_samples_per_dataset is not None:
        rng = random.Random(cfg.dataset.synthetic_split_seed)
        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in out:
            by_dataset[row["dataset_name"]].append(row)
        out = []
        for dataset_rows in by_dataset.values():
            rng.shuffle(dataset_rows)
            out.extend(dataset_rows[: cfg.max_samples_per_dataset])
    if cfg.dataset.sample_limit is not None and len(out) > cfg.dataset.sample_limit:
        rng = random.Random(cfg.dataset.synthetic_split_seed)
        rng.shuffle(out)
        out = out[: cfg.dataset.sample_limit]
    return sorted(out, key=lambda row: (row["dataset_name"], row["task_name"], row["split"], row["sample_id"]))


def _dense_acts(sae, x: torch.Tensor) -> torch.Tensor:
    if hasattr(sae, "threshold") and hasattr(sae, "pre_acts"):
        pre_acts = sae.pre_acts(x)
        threshold = sae.threshold.to(device=pre_acts.device, dtype=pre_acts.dtype)
        return torch.relu(pre_acts) * (pre_acts > threshold).to(pre_acts.dtype)
    acts, indices = sae.encode(x)
    if indices is None:
        return acts
    dense = x.new_zeros((*acts.shape[:-1], sae.num_latents))
    dense.scatter_(-1, indices, acts)
    return dense


def _aggregate_sample(
    sae,
    residual: torch.Tensor,
    token_roles: torch.Tensor,
    *,
    stats,
    cfg: ResidualFeatureDiscoveryConfig,
    device: str,
) -> torch.Tensor | None:
    mask = _role_mask(token_roles, cfg.token_selection)
    if not bool(mask.any()):
        return None
    selected = residual[mask].to(device=device, dtype=torch.float32)
    selected = stats.normalize(selected) if stats is not None else selected

    if cfg.aggregation == "max":
        agg = torch.full((sae.num_latents,), -torch.inf, dtype=torch.float32, device=device)
        with torch.inference_mode():
            for start in range(0, selected.shape[0], cfg.batch_size):
                acts = _dense_acts(sae, selected[start : start + cfg.batch_size])
                agg = torch.maximum(agg, acts.max(dim=0).values.float())
                del acts
        agg = torch.where(torch.isfinite(agg), agg, torch.zeros_like(agg))
        return agg.cpu()

    if cfg.aggregation == "mean":
        total = torch.zeros((sae.num_latents,), dtype=torch.float32, device=device)
        n_tokens = 0
        with torch.inference_mode():
            for start in range(0, selected.shape[0], cfg.batch_size):
                acts = _dense_acts(sae, selected[start : start + cfg.batch_size])
                total += acts.float().sum(dim=0)
                n_tokens += acts.shape[0]
                del acts
        assert n_tokens > 0, "Expected at least one selected token"
        return (total / n_tokens).cpu()

    raise ValueError(f"Unsupported aggregation: {cfg.aggregation!r}")


def _update_stats(stats: _GroupStats | None, agg: torch.Tensor) -> _GroupStats:
    if stats is None:
        return _GroupStats(n=1, act_sum=agg.double(), active_sum=(agg > 0).double())
    stats.n += 1
    stats.act_sum += agg.double()
    stats.active_sum += (agg > 0).double()
    return stats


def _top_feature_rows(
    *,
    task_key: str,
    label: str,
    labels: list[str],
    direction: str,
    feature_i: torch.Tensor,
    score: torch.Tensor,
    pos_mean: torch.Tensor,
    neg_mean: torch.Tensor,
    pos_active: torch.Tensor,
    neg_active: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    sign = 1 if direction == "label_up" else -1
    dataset_name, task_name, field_name = task_key.split("::", maxsplit=2)
    for rank, feat in enumerate(feature_i.tolist(), start=1):
        rows.append(
            {
                "candidate_id": f"{dataset_name}.{task_name}.{field_name}.{label}.{direction}.{rank:03d}.f{feat}",
                "dataset_name": dataset_name,
                "task_name": task_name,
                "field_name": field_name,
                "labels": labels,
                "label": label,
                "direction": direction,
                "suggested_sign": sign,
                "feature_idx": int(feat),
                "rank": rank,
                "score": float(score[feat].item()),
                "pos_mean": float(pos_mean[feat].item()),
                "neg_mean": float(neg_mean[feat].item()),
                "pos_active_rate": float(pos_active[feat].item()),
                "neg_active_rate": float(neg_active[feat].item()),
            }
        )
    return rows


def discover_residual_features(cfg: ResidualFeatureDiscoveryConfig) -> dict[str, Any]:
    device = resolve_visible_device(cfg.device)
    cache_dir = Path(cfg.residual_cache_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _filter_manifest_rows(_read_jsonl(cache_dir / "manifest.jsonl"), cfg)
    assert rows, f"No manifest rows matched discovery config in {cache_dir}"

    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)
    totals: dict[str, _GroupStats] = {}
    label_stats: dict[tuple[str, str], _GroupStats] = {}
    task_labels: dict[str, list[str]] = {}
    counts = Counter((row["dataset_name"], row["split"]) for row in rows)
    logger.info("Starting feature discovery over %d samples: %s", len(rows), dict(counts))

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)

    n_used = 0
    for shard_name, shard_rows in sorted(by_shard.items()):
        shard = load_file(str(cache_dir / shard_name))
        residual = shard["residual"]
        token_roles = shard["token_roles"]
        for row in sorted(shard_rows, key=lambda item: int(item["offset"])):
            offset = int(row["offset"])
            length = int(row["length"])
            label = _row_gold_label(row)
            if label is None:
                continue
            agg = _aggregate_sample(
                sae,
                residual[offset : offset + length],
                token_roles[offset : offset + length],
                stats=stats,
                cfg=cfg,
                device=device,
            )
            if agg is None:
                continue
            task_key = _row_task_key(row)
            task_labels[task_key] = list(row["schema"]["primary_field"]["labels"])
            totals[task_key] = _update_stats(totals.get(task_key), agg)
            label_stats[(task_key, label)] = _update_stats(label_stats.get((task_key, label)), agg)
            n_used += 1
        logger.info("Processed shard=%s rows=%d used=%d", shard_name, len(shard_rows), n_used)

    rankings: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    for task_key, total in sorted(totals.items()):
        labels = task_labels[task_key]
        rankings[task_key] = {"labels": labels, "n_samples": total.n, "label_rankings": {}}
        for label in labels:
            pos = label_stats.get((task_key, label))
            if pos is None or pos.n < cfg.min_positive_samples or total.n - pos.n < cfg.min_positive_samples:
                continue
            neg_n = total.n - pos.n
            pos_mean = pos.act_sum / pos.n
            neg_mean = (total.act_sum - pos.act_sum) / neg_n
            pos_active = pos.active_sum / pos.n
            neg_active = (total.active_sum - pos.active_sum) / neg_n
            score = pos_mean - neg_mean
            k = min(cfg.top_k_per_label, score.numel())
            up_i = torch.topk(score, k=k).indices
            down_i = torch.topk(-score, k=k).indices
            up_rows = _top_feature_rows(
                task_key=task_key,
                label=label,
                labels=labels,
                direction="label_up",
                feature_i=up_i,
                score=score,
                pos_mean=pos_mean,
                neg_mean=neg_mean,
                pos_active=pos_active,
                neg_active=neg_active,
            )
            down_rows = _top_feature_rows(
                task_key=task_key,
                label=label,
                labels=labels,
                direction="label_down",
                feature_i=down_i,
                score=score,
                pos_mean=pos_mean,
                neg_mean=neg_mean,
                pos_active=pos_active,
                neg_active=neg_active,
            )
            rankings[task_key]["label_rankings"][label] = {
                "n_positive": pos.n,
                "n_negative": neg_n,
                "top_label_up": up_rows,
                "top_label_down": down_rows,
            }
            candidates.extend(up_rows)
            candidates.extend(down_rows)

    metadata = {
        "config": cfg.to_dict(),
        "n_manifest_rows": len(rows),
        "n_used_samples": n_used,
        "num_latents": int(sae.num_latents),
        "tasks": sorted(rankings),
    }
    with (output_dir / "feature_rankings.json").open("w") as fd:
        json.dump({"metadata": metadata, "rankings": rankings}, fd, indent=2)
    with (output_dir / "steering_candidates.jsonl").open("w") as fd:
        for row in candidates:
            fd.write(json.dumps(row) + "\n")
    with (output_dir / "summary.json").open("w") as fd:
        json.dump(metadata | {"n_candidates": len(candidates)}, fd, indent=2)
    logger.info("Feature discovery complete: used=%d candidates=%d output_dir=%s", n_used, len(candidates), output_dir)
    return metadata | {"n_candidates": len(candidates), "output_dir": str(output_dir)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ResidualFeatureDiscoveryConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    discover_residual_features(cfg)


if __name__ == "__main__":
    main()
