

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from simple_parsing import parse, Serializable, list_field

from gemma3_experiments.residual_sae.normalization import load_residual_stats

from ..models import load_residual_sae

logger = logging.getLogger(__name__)

# Token role constants (match cache_dense output)
ROLE_PROMPT = 0
ROLE_IMAGE = 1
ROLE_GENERATED = 2


@dataclass
class FeatureDiscoveryConfig(Serializable):
    residual_cache_dir: str = "./qwen_residual_sae_cache/residual"
    residual_sae_path: str = "./qwen_residual_sae_ckpts/best_finite"
    output_dir: str = "./qwen_feature_discovery"
    device: str = "cuda:0"
    token_selection: str = "all"  # all | image | generated | image_and_generated | prompt
    aggregation: str = "max"       # max | mean
    batch_size: int = 512
    top_k_per_label: int = 50
    max_samples_per_dataset: int | None = None
    splits: list[str] = list_field("train")


# Positive label names per task (last label in schema = positive class)
POSITIVE_LABELS: dict[str, str] = {
    "hateful_memes_binary": "hateful",
    "subtask_a_hate_speech": "Hate Speech",
    "misogyny_and_subtypes": "misogynous",
    "offensive_detection": "offensive",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _role_mask(roles: torch.Tensor, selection: str) -> torch.Tensor:
    if selection == "all":
        return torch.ones_like(roles, dtype=torch.bool)
    elif selection == "image":
        return roles == ROLE_IMAGE
    elif selection == "generated":
        return roles == ROLE_GENERATED
    elif selection == "image_and_generated":
        return (roles == ROLE_IMAGE) | (roles == ROLE_GENERATED)
    elif selection == "prompt":
        return roles == ROLE_PROMPT
    raise ValueError(f"Unknown token_selection: {selection!r}")


def _get_sample_acts(sae, residual: torch.Tensor, residual_stats, aggregation: str) -> torch.Tensor:
    """Return (n_latents,) activation vector for a single sample."""
    sae_in = residual_stats.normalize(residual) if residual_stats is not None else residual
    acts, _ = sae.encode(sae_in)  # (n_tokens, n_latents)
    if aggregation == "max":
        return acts.max(dim=0).values
    else:
        return acts.mean(dim=0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FeatureDiscoveryConfig)

    cache_dir = Path(cfg.residual_cache_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device

    # Load SAE and stats
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    sae.eval()
    residual_stats = load_residual_stats(cfg.residual_sae_path)
    n_latents = sae.num_latents
    logger.info("SAE loaded: %d latents from %s", n_latents, cfg.residual_sae_path)

    # Load manifest
    manifest_rows = _read_jsonl(cache_dir / "manifest.jsonl")
    splits = set(cfg.splits)
    manifest_rows = [r for r in manifest_rows if r.get("split") in splits]
    logger.info("Manifest: %d rows in splits %s", len(manifest_rows), cfg.splits)

    # Group by task
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        task_key = f"{row['dataset_name']}::{row['task_name']}"
        by_task[task_key].append(row)

    all_candidates = []

    for task_key, rows in sorted(by_task.items()):
        ds_name, task_name = task_key.split("::", 1)
        pos_label = POSITIVE_LABELS.get(task_name)
        if pos_label is None:
            logger.info("Skipping multiclass task %s (no binary positive label defined)", task_key)
            continue

        pos_acts_list: list[torch.Tensor] = []
        neg_acts_list: list[torch.Tensor] = []
        n_pos = n_neg = 0

        # Group rows by shard
        by_shard: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if cfg.max_samples_per_dataset and (n_pos + n_neg) >= cfg.max_samples_per_dataset:
                break
            by_shard[row["shard_file"]].append(row)

        for shard_name, shard_rows in sorted(by_shard.items()):
            shard_path = cache_dir / shard_name
            if not shard_path.exists():
                logger.warning("Shard not found: %s", shard_path)
                continue
            try:
                tensors = load_file(str(shard_path))
            except Exception as e:
                logger.warning("Failed to load %s: %s", shard_path, e)
                continue

            # Residual shards contain only 'residual' (n_tokens, d_model).
            # Token roles are not stored in residual shards; skip role filtering.
            residual_all = tensors["residual"].to(device=device, dtype=torch.float32)

            for row in shard_rows:
                offset = row["offset"]
                length = row["length"]
                sample_residual = residual_all[offset: offset + length]

                # Token selection not available for residual shards; use all tokens.
                if sample_residual.shape[0] == 0:
                    continue
                selected = sample_residual

                with torch.no_grad():
                    acts = _get_sample_acts(sae, selected, residual_stats, cfg.aggregation)

                gold = row.get("gold_fields", {})
                gold_val = str(list(gold.values())[0]).lower() if gold else None
                is_positive = (gold_val == pos_label.lower()) if gold_val else None
                if is_positive is None:
                    continue
                if is_positive:
                    pos_acts_list.append(acts.cpu())
                    n_pos += 1
                else:
                    neg_acts_list.append(acts.cpu())
                    n_neg += 1

        if n_pos < 5 or n_neg < 5:
            logger.info("Task %s: too few samples (pos=%d, neg=%d), skipping", task_key, n_pos, n_neg)
            continue

        logger.info("Task %s: n_pos=%d n_neg=%d", task_key, n_pos, n_neg)

        # Stack and compute mean activations per class
        pos_stack = torch.stack(pos_acts_list)   # (n_pos, n_latents)
        neg_stack = torch.stack(neg_acts_list)   # (n_neg, n_latents)
        pos_mean = pos_stack.mean(dim=0)
        neg_mean = neg_stack.mean(dim=0)
        pos_rate = (pos_stack > 0).float().mean(dim=0)
        neg_rate = (neg_stack > 0).float().mean(dim=0)

        # Score: mean activation difference, normalized by max
        diff = pos_mean - neg_mean  # positive = fires more on hateful
        # Top-k "label_up" features (fire more on positive)
        top_up_vals, top_up_idx = diff.topk(cfg.top_k_per_label)
        # Top-k "label_down" features (fire more on negative)
        top_down_vals, top_down_idx = (-diff).topk(cfg.top_k_per_label)

        for rank, (idx, val) in enumerate(zip(top_up_idx.tolist(), top_up_vals.tolist())):
            all_candidates.append({
                "dataset_name": ds_name,
                "task_name": task_name,
                "positive_label": pos_label,
                "direction": "label_up",
                "rank": rank,
                "feature_idx": idx,
                "mean_diff": round(val, 6),
                "pos_mean_act": round(float(pos_mean[idx]), 6),
                "neg_mean_act": round(float(neg_mean[idx]), 6),
                "pos_fire_rate": round(float(pos_rate[idx]), 4),
                "neg_fire_rate": round(float(neg_rate[idx]), 4),
                "n_pos": n_pos,
                "n_neg": n_neg,
            })
        for rank, (idx, val) in enumerate(zip(top_down_idx.tolist(), top_down_vals.tolist())):
            all_candidates.append({
                "dataset_name": ds_name,
                "task_name": task_name,
                "positive_label": pos_label,
                "direction": "label_down",
                "rank": rank,
                "feature_idx": idx,
                "mean_diff": round(-val, 6),  # negative = fires more on negative class
                "pos_mean_act": round(float(pos_mean[idx]), 6),
                "neg_mean_act": round(float(neg_mean[idx]), 6),
                "pos_fire_rate": round(float(pos_rate[idx]), 4),
                "neg_fire_rate": round(float(neg_rate[idx]), 4),
                "n_pos": n_pos,
                "n_neg": n_neg,
            })

    # Write output
    out_path = output_dir / "feature_candidates.jsonl"
    with out_path.open("w") as fd:
        for row in all_candidates:
            fd.write(json.dumps(row) + "\n")
    logger.info("Wrote %d feature candidates to %s", len(all_candidates), out_path)

    # Summary: top-5 features per task/direction
    summary_path = output_dir / "feature_summary.txt"
    with summary_path.open("w") as fd:
        by_task_dir: dict[tuple, list] = defaultdict(list)
        for c in all_candidates:
            by_task_dir[(c["dataset_name"], c["task_name"], c["direction"])].append(c)
        for (ds, task, direction), candidates in sorted(by_task_dir.items()):
            top5 = candidates[:5]
            fd.write(f"\n{ds}/{task} [{direction}] (pos_label={top5[0]['positive_label']}, n_pos={top5[0]['n_pos']}, n_neg={top5[0]['n_neg']})\n")
            for c in top5:
                fd.write(f"  feature={c['feature_idx']:6d}  diff={c['mean_diff']:+.4f}  "
                         f"pos_rate={c['pos_fire_rate']:.3f}  neg_rate={c['neg_fire_rate']:.3f}\n")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
