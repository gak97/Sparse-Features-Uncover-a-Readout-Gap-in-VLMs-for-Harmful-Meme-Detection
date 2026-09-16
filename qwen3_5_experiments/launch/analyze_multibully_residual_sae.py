"""Matched pilot residual-SAE probe ablation on the locked MultiBully split."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from simple_parsing import Serializable, parse

from gemma3_experiments.residual_sae.normalization import load_residual_stats

from ..models import load_residual_sae
from .analyze_exist_multilingual import ROLE_NAMES, _read_jsonl
from .analyze_multibully import _fit_bilinear, _fit_linear, _labels, _top_features


logger = logging.getLogger(__name__)


@dataclass
class MultiBullyResidualAnalysisConfig(Serializable):
    residual_cache_dir: str = "./outputs/multibully/residual_raw"
    dense_cache_dir: str = "./outputs/multibully/dense_raw"
    residual_sae_path: str = "./outputs/qwen_residual_sae_ckpts/best_finite"
    output_dir: str = "./outputs/multibully/residual_analysis_raw_ocr"
    device: str = "cuda:0"
    C: float = 1.0
    max_iter: int = 3000
    top_k_features: int = 256
    bilinear_rank: int = 16
    bilinear_epochs: int = 200
    bilinear_lr: float = 1e-3
    bilinear_weight_decay: float = 1e-3
    bilinear_patience: int = 20
    feature_top_k: int = 50


def _dense_shard_name(residual_shard: str) -> str:
    if not residual_shard.startswith("residual_"):
        raise ValueError(f"Unexpected residual shard name: {residual_shard}")
    return f"dense_{residual_shard.removeprefix('residual_')}"


def _extract_features(sae, stats, residual_dir: Path, dense_dir: Path, rows: list[dict], device: str) -> tuple[list[dict], dict[str, np.ndarray]]:
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)
    vectors: dict[str, list[np.ndarray]] = {name: [] for name in ROLE_NAMES}
    metadata: list[dict] = []
    for residual_shard, shard_rows in sorted(by_shard.items()):
        residual_all = load_file(str(residual_dir / residual_shard))["residual"].to(device=device, dtype=torch.float32)
        roles_all = load_file(str(dense_dir / _dense_shard_name(residual_shard)))["token_roles"].to(device)
        for row in shard_rows:
            offset, length = int(row["offset"]), int(row["length"])
            residual = residual_all[offset: offset + length]
            if not len(residual):
                continue
            sae_input = stats.normalize(residual) if stats is not None else residual
            with torch.no_grad():
                acts, _ = sae.encode(sae_input)
            roles = roles_all[offset: offset + length]
            pooled: dict[str, torch.Tensor] = {"all": acts.max(dim=0).values}
            for name, role in ROLE_NAMES.items():
                if role is None:
                    continue
                role_acts = acts[roles == role]
                pooled[name] = role_acts.max(dim=0).values if len(role_acts) else torch.zeros(sae.num_latents, device=device)
            for name, value in pooled.items():
                vectors[name].append(value.cpu().float().numpy())
            metadata.append(row)
    return metadata, {name: np.asarray(values, dtype=np.float32) for name, values in vectors.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(MultiBullyResidualAnalysisConfig)
    residual_dir, dense_dir, output_dir = Path(cfg.residual_cache_dir), Path(cfg.dense_cache_dir), Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [row for row in _read_jsonl(residual_dir / "manifest.jsonl") if row.get("task_name") == "cyberbullying_detection"]
    sae = load_residual_sae(cfg.residual_sae_path, device=cfg.device).eval()
    stats = load_residual_stats(cfg.residual_sae_path)
    metadata, features = _extract_features(sae, stats, residual_dir, dense_dir, manifest, cfg.device)
    labels = _labels(metadata)
    train = np.asarray([index for index, row in enumerate(metadata) if row["split"] == "probe_train"], dtype=np.int64)
    evaluation = np.asarray([index for index, row in enumerate(metadata) if row["split"] == "eval"], dtype=np.int64)
    if not len(train) or len(evaluation) != 1000:
        raise ValueError(f"Expected non-empty probe train and locked 1,000-example eval split, found {len(train)}/{len(evaluation)}")

    results: list[dict] = []
    top_features: dict[str, dict] = {}
    for role, matrix in features.items():
        metric, classifier, _scaler, _prediction = _fit_linear(matrix[train], labels[train], matrix[evaluation], labels[evaluation], cfg)
        results.append({"representation": role, "model": "linear_residual_sae", "n_train": int(len(train)), "n_eval": int(len(evaluation)), **metric})
        top_features[f"MultiBully::cyberbullying_detection::{role}"] = _top_features(classifier, cfg.feature_top_k)
    bilinear = _fit_bilinear(
        features["image"][train], features["prompt"][train], labels[train],
        features["image"][evaluation], features["prompt"][evaluation], labels[evaluation], cfg,
    )
    results.append({"representation": "image_x_prompt", "model": "low_rank_bilinear_residual_sae", "n_train": int(len(train)), "n_eval": int(len(evaluation)), **bilinear})
    with (output_dir / "probe_results.jsonl").open("w") as fd:
        for row in results:
            fd.write(json.dumps(row) + "\n")
    (output_dir / "top_features.json").write_text(json.dumps(top_features, indent=2) + "\n")
    (output_dir / "analysis_summary.json").write_text(json.dumps({
        "residual_cache_dir": str(residual_dir), "residual_sae_path": cfg.residual_sae_path,
        "n_probe_train": int(len(train)), "n_eval": int(len(evaluation)),
    }, indent=2) + "\n")
    logger.info("Wrote MultiBully residual-SAE probe results to %s", output_dir)


if __name__ == "__main__":
    main()
