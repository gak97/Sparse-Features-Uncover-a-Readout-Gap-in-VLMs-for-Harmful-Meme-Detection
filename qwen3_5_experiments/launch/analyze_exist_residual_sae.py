

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
from .analyze_exist_multilingual import (
    ROLE_NAMES,
    _fit_bilinear,
    _fit_linear,
    _indices,
    _label,
    _read_jsonl,
    _top_features,
)

logger = logging.getLogger(__name__)


@dataclass
class ExistResidualAnalysisConfig(Serializable):
    residual_cache_dir: str = "./outputs/exist_multilingual/residual_raw"
    dense_cache_dir: str = "./outputs/exist_multilingual/dense_raw"
    residual_sae_path: str = "./outputs/qwen_residual_sae_ckpts/best_finite"
    output_dir: str = "./outputs/exist_multilingual/residual_analysis_raw_ocr"
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


def _dense_shard_name(residual_shard_name: str) -> str:
    if not residual_shard_name.startswith("residual_"):
        raise ValueError(f"Unexpected residual shard name: {residual_shard_name}")
    return f"dense_{residual_shard_name.removeprefix('residual_')}"


def _extract_residual_features(
    sae, residual_stats, residual_dir: Path, dense_dir: Path, rows: list[dict], device: str
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Encode residual states and use the paired dense cache solely for roles."""
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)

    vectors: dict[str, list[np.ndarray]] = {name: [] for name in ROLE_NAMES}
    metadata: list[dict] = []
    for residual_shard, shard_rows in sorted(by_shard.items()):
        residual_all = load_file(str(residual_dir / residual_shard))["residual"].to(device=device, dtype=torch.float32)
        dense_all = load_file(str(dense_dir / _dense_shard_name(residual_shard)))
        roles_all = dense_all["token_roles"].to(device)
        for row in shard_rows:
            offset, length = int(row["offset"]), int(row["length"])
            residual = residual_all[offset: offset + length]
            if not len(residual):
                continue
            sae_input = residual_stats.normalize(residual) if residual_stats is not None else residual
            with torch.no_grad():
                acts, _ = sae.encode(sae_input)
            roles = roles_all[offset: offset + length]
            pooled: dict[str, torch.Tensor] = {"all": acts.max(dim=0).values}
            for name, role in ROLE_NAMES.items():
                if role is None:
                    continue
                role_acts = acts[roles == role]
                pooled[name] = role_acts.max(dim=0).values if len(role_acts) else torch.zeros(sae.num_latents, device=device)
            for name, vector in pooled.items():
                vectors[name].append(vector.cpu().float().numpy())
            metadata.append(row)
    if not metadata:
        raise ValueError("No usable residual EXIST samples")
    return metadata, {name: np.asarray(values, dtype=np.float32) for name, values in vectors.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ExistResidualAnalysisConfig)
    residual_dir, dense_dir, output_dir = Path(cfg.residual_cache_dir), Path(cfg.dense_cache_dir), Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = [row for row in _read_jsonl(residual_dir / "manifest.jsonl") if row.get("task_name") == "sexism_detection"]
    if {row.get("split") for row in manifest} != {"probe_train", "eval"}:
        raise ValueError("Residual EXIST manifest must contain fixed probe_train and eval partitions")

    sae = load_residual_sae(cfg.residual_sae_path, device=cfg.device).eval()
    residual_stats = load_residual_stats(cfg.residual_sae_path)
    metadata, features = _extract_residual_features(sae, residual_stats, residual_dir, dense_dir, manifest, cfg.device)
    labels = np.asarray([_label(row) for row in metadata], dtype=np.int32)
    train_en, train_es = _indices(metadata, "probe_train", "en"), _indices(metadata, "probe_train", "es")
    eval_en, eval_es = _indices(metadata, "eval", "en"), _indices(metadata, "eval", "es")
    conditions = {
        "en_to_es": (train_en, eval_es),
        "es_to_en": (train_es, eval_en),
        "bilingual_to_en": (np.concatenate([train_en, train_es]), eval_en),
        "bilingual_to_es": (np.concatenate([train_en, train_es]), eval_es),
    }

    results: list[dict] = []
    top_features: dict[str, dict] = {}
    for condition, (train_index, evaluation_index) in conditions.items():
        logger.info("=== %s: train=%d eval=%d ===", condition, len(train_index), len(evaluation_index))
        for role, matrix in features.items():
            metric, clf, _scaler, _prediction = _fit_linear(
                matrix[train_index], labels[train_index], matrix[evaluation_index], labels[evaluation_index], cfg
            )
            results.append({
                "condition": condition, "representation": role, "model": "linear_residual_sae",
                "n_train": int(len(train_index)), "n_eval": int(len(evaluation_index)), **metric,
            })
            top_features[f"{condition}::{role}"] = _top_features(clf, cfg.feature_top_k)
        bilinear = _fit_bilinear(
            features["image"][train_index], features["prompt"][train_index], labels[train_index],
            features["image"][evaluation_index], features["prompt"][evaluation_index], labels[evaluation_index], cfg,
        )
        results.append({
            "condition": condition, "representation": "image_x_prompt", "model": "low_rank_bilinear_residual_sae",
            "n_train": int(len(train_index)), "n_eval": int(len(evaluation_index)), **bilinear,
        })

    with (output_dir / "transfer_results.jsonl").open("w") as fd:
        for row in results:
            fd.write(json.dumps(row) + "\n")
    (output_dir / "top_features.json").write_text(json.dumps(top_features, indent=2) + "\n")
    (output_dir / "analysis_summary.json").write_text(json.dumps({
        "residual_cache_dir": str(residual_dir), "residual_sae_path": cfg.residual_sae_path,
        "n_samples": len(metadata), "n_probe_train_en": len(train_en), "n_probe_train_es": len(train_es),
        "n_eval_en": len(eval_en), "n_eval_es": len(eval_es),
    }, indent=2) + "\n")
    logger.info("Wrote residual-SAE EXIST transfer results to %s", output_dir)


if __name__ == "__main__":
    main()