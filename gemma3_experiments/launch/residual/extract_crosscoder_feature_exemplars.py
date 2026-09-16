

import heapq
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.probe_fhm_crosscoder import (
    DenseLayerSpec,
    DenseShardReader,
    GemmaCrosscoder,
    _manifest_key,
    _role_mask,
)
from gemma3_experiments.residual_sae import resolve_visible_device

logger = logging.getLogger(__name__)


@dataclass
class CrosscoderExemplarConfig:
    # Feature source: either a bilinear_feature_indices.json path or explicit list
    bilinear_feature_indices_path: str = ""      # JSON with {"feature_indices": [...]}
    extra_feature_indices: list[int] = field(default_factory=list)

    # Dense cache
    dense_cache_dir: str = ""
    crosscoder_repo_id: str = "google/gemma-scope-2-12b-it"
    crosscoder_hookpoint: str = "resid_post/layer_31_width_65k_l0_medium"
    layer_name: str = "layer31"

    # Which rows to scan
    datasets: list[str] = field(default_factory=lambda: ["Hateful_Memes_Dataset"])
    splits: list[str] = field(default_factory=lambda: ["train", "validation", "val", "dev"])

    # Extraction settings
    token_selection: str = "all"
    top_k_examples: int = 20
    batch_size: int = 64
    chunk_size: int = 64
    shard_cache_size: int = 16

    output_path: str = ""
    device: str = "cuda:0"


def _load_feature_indices(cfg: CrosscoderExemplarConfig) -> list[int]:
    feat_set: set[int] = set(cfg.extra_feature_indices)
    if cfg.bilinear_feature_indices_path:
        p = Path(cfg.bilinear_feature_indices_path)
        assert p.exists(), f"bilinear_feature_indices_path not found: {p}"
        raw = json.loads(p.read_text())
        if isinstance(raw, list):
            feat_set.update(int(x) for x in raw)
        elif isinstance(raw, dict) and "feature_indices" in raw:
            feat_set.update(int(x) for x in raw["feature_indices"])
        else:
            raise ValueError(f"Unexpected format in {p}: {list(raw)[:5]}")
    assert feat_set, "No feature indices supplied — use --bilinear_feature_indices_path or --extra_feature_indices"
    return sorted(feat_set)


def _d_model_from_shard(cache_dir: Path) -> int:
    shards = sorted(cache_dir.glob("*.safetensors"))
    assert shards, f"No safetensors shards in {cache_dir}"
    with safe_open(str(shards[0]), framework="pt", device="cpu") as fd:
        return int(fd.get_tensor("hidden").shape[1])


def _manifest_rows(cache_dir: Path, datasets: list[str], splits: list[str]) -> list[dict[str, Any]]:
    rows = []
    for line in open(cache_dir / "manifest.jsonl"):
        r = json.loads(line)
        if datasets and r.get("dataset_name") not in datasets:
            continue
        if splits and r.get("split") not in splits:
            continue
        r["_cache_dir"] = str(cache_dir)
        rows.append(r)
    return rows


def run(cfg: CrosscoderExemplarConfig) -> None:
    device = resolve_visible_device(cfg.device)
    cache_dir = Path(cfg.dense_cache_dir)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Feature indices ────────────────────────────────────────────────────────
    feat_indices = _load_feature_indices(cfg)
    feat_set = set(feat_indices)
    # local position in feat_indices list for fast update
    feat_to_pos = {fid: i for i, fid in enumerate(feat_indices)}
    K = len(feat_indices)
    logger.info("Extracting exemplars for %d features", K)

    # ── Load crosscoder SAE ────────────────────────────────────────────────────
    d_model = _d_model_from_shard(cache_dir)
    layer_specs = [DenseLayerSpec(cfg.layer_name, str(cache_dir), str(cache_dir))]
    logger.info("Loading crosscoder: %s  %s", cfg.crosscoder_repo_id, cfg.crosscoder_hookpoint)
    crosscoder = GemmaCrosscoder.load_from_hub(
        cfg.crosscoder_repo_id, cfg.crosscoder_hookpoint,
        n_layers=len(layer_specs), d_model=d_model, device=device,
    )
    logger.info("Crosscoder num_latents=%d", crosscoder.num_latents)

    # ── Load manifest rows ─────────────────────────────────────────────────────
    rows = _manifest_rows(cache_dir, cfg.datasets, cfg.splits)
    logger.info("Scanning %d rows (datasets=%s splits=%s)", len(rows), cfg.datasets, cfg.splits)

    # Min-heap per feature: (neg_activation, counter, example_dict)
    # counter breaks ties so dicts are never compared directly.
    heaps: list[list[tuple[float, int, dict[str, Any]]]] = [[] for _ in range(K)]
    _counter = 0

    reader = DenseShardReader(layer_specs, cfg.shard_cache_size, cfg.token_selection)

    for idx, row in enumerate(rows):
        if (idx + 1) % 500 == 0:
            logger.info("  scanned %d / %d rows", idx + 1, len(rows))

        try:
            hidden_tld, roles = reader.get_sample([row])
        except Exception as exc:
            logger.debug("Skipping row %s: %s", _manifest_key(row), exc)
            continue

        mask = _role_mask(roles, cfg.token_selection)
        if not mask.any():
            continue
        sel = hidden_tld[mask]  # [T, d]

        # Encode in chunks
        chunks = []
        for s in range(0, sel.shape[0], cfg.chunk_size):
            chunk_acts = crosscoder.encode(sel[s: s + cfg.chunk_size])  # [t, num_latents] (may be on GPU)
            if isinstance(chunk_acts, torch.Tensor):
                chunk_acts = chunk_acts.cpu().numpy()
            chunks.append(chunk_acts)
        if not chunks:
            continue
        all_acts = np.concatenate(chunks, axis=0)  # [T, num_latents]

        # Max-pool over tokens for each feature index we care about
        max_per_feature = all_acts[:, feat_indices].max(axis=0)  # [K]

        # Build example dict from manifest metadata
        gf = row.get("gold_fields") or {}
        label = gf.get("label", "")
        ex = {
            "meme_text": row.get("meme_text", ""),
            "gold_label": label,
            "dataset_name": row.get("dataset_name", ""),
            "split": row.get("split", ""),
            "sample_id": row.get("sample_id", ""),
            "generated_text": (row.get("generated_text") or "")[:200],
        }

        for ki, act_val in enumerate(max_per_feature):
            act_f = float(act_val)
            if act_f == 0.0:
                continue
            heap = heaps[ki]
            if len(heap) < cfg.top_k_examples:
                heapq.heappush(heap, (act_f, _counter, ex))
                _counter += 1
            elif act_f > heap[0][0]:
                heapq.heapreplace(heap, (act_f, _counter, ex))
                _counter += 1

    # ── Assemble output ────────────────────────────────────────────────────────
    output: dict[str, list[dict[str, Any]]] = {}
    for ki, fid in enumerate(feat_indices):
        heap = heaps[ki]
        sorted_examples = sorted(heap, key=lambda x: -x[0])
        output[str(fid)] = [
            {**ex_dict, "activation": round(act_val, 5)}
            for act_val, _cnt, ex_dict in sorted_examples
        ]

    output_path.write_text(json.dumps(output, indent=2))
    n_with_examples = sum(1 for v in output.values() if v)
    logger.info("Wrote exemplars for %d / %d features to %s",
                n_with_examples, K, output_path)
    if n_with_examples < K:
        missing = [fid for fid, exs in output.items() if not exs]
        logger.warning("Features with no activating examples: %s", missing[:20])


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(CrosscoderExemplarConfig,
                argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
