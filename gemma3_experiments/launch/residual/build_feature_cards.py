

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.extract_residual_feature_exemplars import (
    ResidualFeatureExemplarConfig,
    run as extract_exemplars,
)
from gemma3_experiments.launch.residual.probe_validate import _read_jsonl

logger = logging.getLogger(__name__)


@dataclass
class FeatureCardConfig:
    # Sources
    top_features_paths: list[str] = field(default_factory=list)
    confounder_overlap_path: str = ""
    recurrent_features_path: str = ""
    extra_feature_indices: list[int] = field(default_factory=list)

    # Exemplar extraction
    residual_sae_path: str = ""
    residual_cache_dir: str = ""
    datasets: list[str] = field(default_factory=lambda: ["Hateful_Memes_Dataset", "MMHS150K", "MAMI", "HarMeme", "Crisis_HateMM", "MultiOFF"])
    splits: list[str] = field(default_factory=lambda: ["train", "validation"])
    token_selection: str = "generated"
    aggregation: str = "max"
    top_k_examples: int = 20
    max_features: int = 200

    # Output
    output_dir: str = ""
    device: str = "cuda:0"


def _load_top_features(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data
    return {}


def _collect_confounder_overlap(path: str) -> dict[int, dict[str, Any]]:
    overlap_path = Path(path)
    if not overlap_path.exists():
        logger.warning("Confounder overlap path missing: %s", overlap_path)
        return {}
    
    by_feature = {}
    # Parse JSONL format: one JSON object per line
    try:
        with open(overlap_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or "feature_idx" not in row:
                    continue
                
                feat_idx = row.get("feature_idx")
                if not isinstance(feat_idx, int):
                    try:
                        feat_idx = int(feat_idx)
                    except (ValueError, TypeError):
                        continue
                
                # Build entry with overlap metadata
                entry = by_feature.setdefault(feat_idx, {"overlap_sources": []})
                
                # Track which (token_selection, source) pairs this feature appears in
                ts = row.get("token_selection", "unknown")
                src = row.get("source", "unknown")
                source_key = f"{ts}:{src}"
                if source_key not in entry["overlap_sources"]:
                    entry["overlap_sources"].append(source_key)
                
                # Store rank and score for sorting
                if "rank" in row:
                    entry.setdefault("ranks", {})[source_key] = row["rank"]
                if "score" in row:
                    entry.setdefault("scores", {})[source_key] = row["score"]
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Error reading confounder overlap file %s: %s", overlap_path, e)
        return {}

    return by_feature


def _build_feature_cards(cfg: FeatureCardConfig) -> list[dict[str, Any]]:
    # Gather feature indices + metadata from all sources
    by_index: dict[int, dict[str, Any]] = {}

    # 1. Confounder overlap
    if cfg.confounder_overlap_path:
        overlap = _collect_confounder_overlap(cfg.confounder_overlap_path)
        for feat_idx, info in overlap.items():
            entry = by_index.setdefault(feat_idx, {"feature_idx": feat_idx, "sources": []})
            entry["confounder_overlap"] = info

    # 2. Recurrent features
    if cfg.recurrent_features_path:
        rec_path = Path(cfg.recurrent_features_path)
        if rec_path.exists():
            rec_data = json.loads(rec_path.read_text())
            if isinstance(rec_data, list):
                for row in rec_data:
                    if not isinstance(row, dict):
                        continue
                    feat_idx = row.get("feature_idx")
                    if feat_idx is not None:
                        entry = by_index.setdefault(int(feat_idx), {"feature_idx": int(feat_idx), "sources": []})
                        entry["recurrent"] = row

    # 3. Top features from probe results
    if cfg.top_features_paths:
        for path_str in cfg.top_features_paths:
            path = Path(path_str)
            if not path.exists():
                continue
            if path.suffix == ".jsonl":
                for row in _read_jsonl(path):
                    feat_idx = row.get("feature_idx")
                    if feat_idx is not None:
                        entry = by_index.setdefault(int(feat_idx), {"feature_idx": int(feat_idx), "sources": []})
                        entry.setdefault("probe_sources", []).append({
                            "path": str(path),
                            "task_key": row.get("task_key"),
                            "rank": row.get("rank"),
                            "weight": row.get("weight"),
                            "direction": row.get("direction"),
                        })
            else:
                data = json.loads(path.read_text())
                # Support two schemas:
                # Schema A (old): {task_key: {class: [feat_dict, ...]}}
                # Schema B (feature_rankings.json):
                #   {rankings: {task_key: {label_rankings: {class: {top_label_up: [...], top_label_down: [...]}}}}}
                if isinstance(data, dict) and "rankings" in data:
                    # Schema B
                    for task_key, task_data in data["rankings"].items():
                        if not isinstance(task_data, dict):
                            continue
                        for cls, cls_data in task_data.get("label_rankings", {}).items():
                            if not isinstance(cls_data, dict):
                                continue
                            for direction in ["top_label_up", "top_label_down"]:
                                for rank, feat in enumerate(cls_data.get(direction, []), 1):
                                    if not isinstance(feat, dict):
                                        continue
                                    feat_idx = feat.get("feature_idx")
                                    if feat_idx is not None:
                                        entry = by_index.setdefault(int(feat_idx), {"feature_idx": int(feat_idx), "sources": []})
                                        entry.setdefault("probe_sources", []).append({
                                            "task_key": task_key,
                                            "class": cls,
                                            "rank": rank,
                                            "direction": direction,
                                            "score": feat.get("score"),
                                            "weight": feat.get("score"),  # expose as weight for display
                                        })
                elif isinstance(data, dict):
                    # Schema A
                    for task_key, by_class in data.items():
                        if not isinstance(by_class, dict):
                            continue
                        for cls, feats in by_class.items():
                            if not isinstance(feats, list):
                                continue
                            for rank, feat in enumerate(feats, 1):
                                if not isinstance(feat, dict):
                                    continue
                                feat_idx = feat.get("feature_idx")
                                if feat_idx is not None:
                                    entry = by_index.setdefault(int(feat_idx), {"feature_idx": int(feat_idx), "sources": []})
                                    entry.setdefault("probe_sources", []).append({
                                        "task_key": task_key,
                                        "class": cls,
                                        "rank": rank,
                                        "weight": feat.get("weight"),
                                        "direction": feat.get("direction"),
                                    })

    # 4. Extra indices — always included regardless of max_features
    extra_set = set(int(idx) for idx in cfg.extra_feature_indices)
    for idx in cfg.extra_feature_indices:
        by_index.setdefault(int(idx), {"feature_idx": int(idx), "sources": []})

    # Apply max_features only to non-extra features so extra indices are never dropped
    auto_features = sorted(
        (v for v in by_index.values() if v["feature_idx"] not in extra_set),
        key=lambda x: x["feature_idx"],
    )[: cfg.max_features]
    extra_features = [by_index[idx] for idx in sorted(extra_set) if idx in by_index]
    selected = sorted(auto_features + extra_features, key=lambda x: x["feature_idx"])
    return selected


def run(cfg: FeatureCardConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect feature indices and metadata
    features = _build_feature_cards(cfg)
    logger.info("Building feature cards for %d features.", len(features))

    # 2. Extract exemplars
    exemplar_cfg = ResidualFeatureExemplarConfig(
        residual_sae_path=cfg.residual_sae_path,
        residual_cache_dir=cfg.residual_cache_dir,
        output_dir=str(output_dir / "exemplars"),
        extra_feature_indices=[str(f["feature_idx"]) for f in features],
        datasets=cfg.datasets,
        splits=cfg.splits,
        token_selection=cfg.token_selection,
        aggregation=cfg.aggregation,
        top_k_examples=cfg.top_k_examples,
        # Use total selected count so extra_feature_indices are not truncated
        max_features=len(features),
        device=cfg.device,
    )
    extract_exemplars(exemplar_cfg)

    # 3. Merge exemplars into feature cards
    exemplar_path = output_dir / "exemplars" / "feature_exemplars.json"
    exemplar_data = {}
    if exemplar_path.exists():
        raw_exemplar = json.loads(exemplar_path.read_text())
        if isinstance(raw_exemplar, dict) and isinstance(raw_exemplar.get("features"), dict):
            exemplar_data = raw_exemplar["features"]
        elif isinstance(raw_exemplar, dict):
            exemplar_data = raw_exemplar

    cards = []
    for feat in features:
        feat_idx = feat["feature_idx"]
        card = {
            "feature_idx": feat_idx,
            "sources": feat.get("sources", []),
            "confounder_overlap": feat.get("confounder_overlap"),
            "recurrent": feat.get("recurrent"),
            "probe_sources": feat.get("probe_sources", []),
            "top_examples": exemplar_data.get(str(feat_idx), []),
        }
        cards.append(card)

    out_path = output_dir / "feature_cards.json"
    out_path.write_text(json.dumps({"features": cards}, indent=2))
    logger.info("Wrote %d feature cards to %s", len(cards), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FeatureCardConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()