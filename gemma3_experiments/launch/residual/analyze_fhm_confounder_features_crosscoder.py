

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
import torch
from safetensors import safe_open
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.analyze_fhm_confounder_features import (
    _build_pairs,
    _feature_stats,
    _load_lr_top_features,
    _plot_overlap,
    _plot_pair_score_hist,
    _plot_scatter,
    _plot_transfer,
    _score_pairs,
    _strip_arrays,
    _transfer_stats,
)
from gemma3_experiments.launch.residual.probe_fhm_crosscoder import (
    DenseLayerSpec,
    DenseShardReader,
    GemmaCrosscoder,
    _aggregate_acts,
    _common_keys,
    _manifest_key,
    _role_mask,
)
from gemma3_experiments.launch.residual.probe_validate import (
    _load_fhm_confounder_metadata,
    _normalise_fhm_id,
    _normalise_group_value,
    _read_jsonl,
    _row_gold_label,
    _row_sample_key,
)
from gemma3_experiments.residual_sae import resolve_visible_device

logger = logging.getLogger(__name__)


@dataclass
class CrosscoderConfounderAuditConfig:
    # Crosscoder
    crosscoder_repo_id: str = "google/gemma-scope-2-4b-it"
    crosscoder_hookpoint: str = "crosscoder/layer_9_17_22_29_width_65k_l0_medium"

    # Multi-layer dense caches (train)
    layer9_train_dense_cache_dir: str = ""
    layer17_train_dense_cache_dir: str = ""
    layer22_train_dense_cache_dir: str = ""
    layer29_train_dense_cache_dir: str = ""

    # Multi-layer dense caches (validation)
    layer9_val_dense_cache_dir: str = ""
    layer17_val_dense_cache_dir: str = ""
    layer22_val_dense_cache_dir: str = ""
    layer29_val_dense_cache_dir: str = ""

    # Layer names shown in log messages
    layer9_name: str = "layer9"
    layer17_name: str = "layer17"
    layer22_name: str = "layer22"
    layer29_name: str = "layer29"

    # IO
    output_dir: str = ""
    fhm_confounder_metadata_path: str = ""

    # Audit settings
    token_selections: list[str] = field(default_factory=lambda: ["all", "prompt_and_image"])
    aggregation: str = "max"
    batch_size: int = 64
    chunk_size: int = 64
    shard_cache_size: int = 16
    top_k_features: int = 100
    transfer_top_k: list[int] = field(default_factory=lambda: [5, 10, 20, 50])

    # Optional comparison files
    # Path to top_features.json from a prior residual-SAE LR probe (to compare overlap).
    compare_lr_top_features_json: str = ""
    # Path to confounder_feature_audit_summary.json from the 65k crosscoder audit run
    # (so the 262k run can report feature-index overlap with the 65k top features).
    compare_crosscoder_audit_json: str = ""

    device: str = "cuda:0"


def _build_layer_specs(cfg: CrosscoderConfounderAuditConfig) -> list[DenseLayerSpec]:
    candidates = [
        (cfg.layer9_name, cfg.layer9_train_dense_cache_dir, cfg.layer9_val_dense_cache_dir),
        (cfg.layer17_name, cfg.layer17_train_dense_cache_dir, cfg.layer17_val_dense_cache_dir),
        (cfg.layer22_name, cfg.layer22_train_dense_cache_dir, cfg.layer22_val_dense_cache_dir),
        (cfg.layer29_name, cfg.layer29_train_dense_cache_dir, cfg.layer29_val_dense_cache_dir),
    ]
    return [DenseLayerSpec(name, train, val) for name, train, val in candidates if train and val]


def _load_fhm_rows_tagged(cache_dir: Path, split_names: set[str]) -> list[dict[str, Any]]:
    """Load FHM rows from a dense cache manifest, tagging each with its cache_dir."""
    rows = []
    for row in _read_jsonl(cache_dir / "manifest.jsonl"):
        if row.get("dataset_name") != "Hateful_Memes_Dataset":
            continue
        if row.get("task_name") != "hateful_memes_binary":
            continue
        if row.get("split") not in split_names:
            continue
        if _row_gold_label(row) not in {"hateful", "not hateful"}:
            continue
        row = dict(row)
        row["_cache_dir"] = str(cache_dir)
        rows.append(row)
    return rows


def _extract_crosscoder_features_for_token_sel(
    rows_by_layer: list[list[dict[str, Any]]],
    key_lists: list[list[str]],
    layer_specs: list[DenseLayerSpec],
    crosscoder: GemmaCrosscoder,
    token_selection: str,
    aggregation: str,
    batch_size: int,
    chunk_size: int,
    shard_cache_size: int,
) -> tuple[list[str], Any]:
    """Extract crosscoder feature vectors for all common-key samples.

    Returns (common_keys, csr_matrix) where each row is the aggregated
    (e.g. max over tokens) sparse crosscoder activation vector.
    """
    common_keys = _common_keys(key_lists)
    row_lookup_by_layer = [{_manifest_key(row): row for row in rows} for rows in rows_by_layer]
    reader = DenseShardReader(layer_specs, shard_cache_size, token_selection)
    feature_rows = []
    for batch_start in range(0, len(common_keys), batch_size):
        batch_keys = common_keys[batch_start : batch_start + batch_size]
        for key in batch_keys:
            sample_rows = [lookup[key] for lookup in row_lookup_by_layer]
            hidden_tld, roles = reader.get_sample(sample_rows)
            mask = _role_mask(roles, token_selection)
            selected = hidden_tld[mask]
            chunks = []
            for start in range(0, selected.shape[0], chunk_size):
                chunks.append(crosscoder.encode(selected[start : start + chunk_size]))
            if chunks:
                acts = torch.cat(chunks, dim=0)
                vec = _aggregate_acts(acts, aggregation)
            else:
                vec = np.zeros((crosscoder.num_latents,), dtype=np.float32)
            feature_rows.append(scipy.sparse.csr_matrix(vec.reshape(1, -1)))
        if (batch_start + batch_size) % 512 == 0 or batch_start + batch_size >= len(common_keys):
            logger.info(
                "  Extracted %d / %d samples (token_sel=%s)",
                min(batch_start + batch_size, len(common_keys)),
                len(common_keys),
                token_selection,
            )
    return common_keys, scipy.sparse.vstack(feature_rows, format="csr", dtype=np.float32)


def _load_prior_crosscoder_top_features(
    audit_json_path: Path,
    token_selections: list[str],
) -> dict[str, set[int]]:
    """Return {token_sel/source_str: set(feature_idx)} for the top-20 of a prior audit."""
    if not audit_json_path.exists():
        return {}
    data = json.loads(audit_json_path.read_text())
    result = {}
    for token_sel, by_source in data.get("audits", {}).items():
        if token_sel not in token_selections:
            continue
        for source, audit in by_source.items():
            key = f"{token_sel}/{source}"
            result[key] = {row["feature_idx"] for row in audit.get("top_features", [])[:20]}
    return result


def run_crosscoder_audit(cfg: CrosscoderConfounderAuditConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(cfg.fhm_confounder_metadata_path)
    assert metadata_path.exists(), f"FHM metadata missing: {metadata_path}"

    layer_specs = _build_layer_specs(cfg)
    assert len(layer_specs) >= 2, "Need at least 2 configured dense layer cache dirs."

    # Load rows per layer (train and val)
    train_rows_by_layer: list[list[dict[str, Any]]] = []
    val_rows_by_layer: list[list[dict[str, Any]]] = []
    train_key_lists: list[list[str]] = []
    val_key_lists: list[list[str]] = []
    for spec in layer_specs:
        train_cache = Path(spec.train_dense_cache_dir)
        val_cache = Path(spec.val_dense_cache_dir)
        assert train_cache.exists(), f"Train cache missing for {spec.name}: {train_cache}"
        assert val_cache.exists(), f"Val cache missing for {spec.name}: {val_cache}"
        train_rows = _load_fhm_rows_tagged(train_cache, {"train"})
        val_rows = _load_fhm_rows_tagged(val_cache, {"validation", "val", "dev"})
        assert train_rows, f"No FHM train rows in {train_cache}"
        assert val_rows, f"No FHM val rows in {val_cache}"
        train_rows_by_layer.append(train_rows)
        val_rows_by_layer.append(val_rows)
        train_key_lists.append([_manifest_key(row) for row in train_rows])
        val_key_lists.append([_manifest_key(row) for row in val_rows])
        logger.info("%s: train=%d val=%d", spec.name, len(train_rows), len(val_rows))

    # Infer d_model from the first shard of the first layer
    first_shard = Path(train_rows_by_layer[0][0]["_cache_dir"]) / train_rows_by_layer[0][0]["shard_file"]
    with safe_open(str(first_shard), framework="pt", device="cpu") as fd:
        d_model = int(fd.get_tensor("hidden").shape[1])

    device = resolve_visible_device(cfg.device)
    logger.info("Loading crosscoder %s/%s (d_model=%d)", cfg.crosscoder_repo_id, cfg.crosscoder_hookpoint, d_model)
    crosscoder = GemmaCrosscoder.load_from_hub(
        cfg.crosscoder_repo_id,
        cfg.crosscoder_hookpoint,
        n_layers=len(layer_specs),
        d_model=d_model,
        device=device,
    )
    logger.info("Crosscoder num_latents=%d", crosscoder.num_latents)

    metadata = _load_fhm_confounder_metadata(metadata_path)

    # Load optional comparison sets
    lr_top_features = (
        _load_lr_top_features(Path(cfg.compare_lr_top_features_json))
        if cfg.compare_lr_top_features_json
        else set()
    )
    prior_crosscoder_features: dict[str, set[int]] = {}
    if cfg.compare_crosscoder_audit_json:
        prior_crosscoder_features = _load_prior_crosscoder_top_features(
            Path(cfg.compare_crosscoder_audit_json),
            cfg.token_selections,
        )
        logger.info("Loaded prior crosscoder top features for %d keys", len(prior_crosscoder_features))

    summary: dict[str, Any] = {
        "config": vars(cfg),
        "crosscoder_hookpoint": cfg.crosscoder_hookpoint,
        "num_latents": crosscoder.num_latents,
        "n_layers_used": len(layer_specs),
        "n_train_rows": len(train_rows_by_layer[0]),
        "n_val_rows": len(val_rows_by_layer[0]),
        "audits": {},
    }

    train_row_by_key = {_manifest_key(row): row for row in train_rows_by_layer[0]}
    val_row_by_key = {_manifest_key(row): row for row in val_rows_by_layer[0]}

    for token_selection in cfg.token_selections:
        logger.info("=== token_selection=%s ===", token_selection)

        train_keys, train_csr = _extract_crosscoder_features_for_token_sel(
            train_rows_by_layer,
            train_key_lists,
            layer_specs,
            crosscoder,
            token_selection,
            cfg.aggregation,
            cfg.batch_size,
            cfg.chunk_size,
            cfg.shard_cache_size,
        )
        val_keys, val_csr = _extract_crosscoder_features_for_token_sel(
            val_rows_by_layer,
            val_key_lists,
            layer_specs,
            crosscoder,
            token_selection,
            cfg.aggregation,
            cfg.batch_size,
            cfg.chunk_size,
            cfg.shard_cache_size,
        )
        train_key_to_idx = {key: idx for idx, key in enumerate(train_keys)}
        val_key_to_idx = {key: idx for idx, key in enumerate(val_keys)}

        summary["audits"][token_selection] = {}
        for source in ["pseudo_img", "pseudo_text"]:
            train_pairs = _build_pairs(
                [train_row_by_key[k] for k in train_keys if k in train_row_by_key],
                metadata,
                set(train_keys),
                source,
            )
            val_pairs = _build_pairs(
                [val_row_by_key[k] for k in val_keys if k in val_row_by_key],
                metadata,
                set(val_keys),
                source,
            )
            audit = _feature_stats(train_csr, train_key_to_idx, train_pairs, cfg.top_k_features)
            audit["n_train_pairs"] = len(train_pairs)
            audit["n_val_pairs"] = len(val_pairs)
            audit["transfer"] = _transfer_stats(
                audit, val_csr, val_key_to_idx, val_pairs, cfg.transfer_top_k
            )
            audit["validation_pair_scores"] = _score_pairs(
                audit, val_csr, val_key_to_idx, val_pairs, 20
            )

            audit_top20 = {row["feature_idx"] for row in audit["top_features"][:20]}

            # Overlap with prior residual-SAE LR probe features
            if lr_top_features:
                audit["overlap_with_lr_fhm_top20"] = {
                    "n_overlap": len(audit_top20 & lr_top_features),
                    "features": sorted(audit_top20 & lr_top_features),
                }

            # Overlap with prior crosscoder audit (e.g. 65k vs 262k)
            prior_key = f"{token_selection}/{source}"
            if prior_key in prior_crosscoder_features:
                prior_set = prior_crosscoder_features[prior_key]
                audit["overlap_with_prior_crosscoder_top20"] = {
                    "n_overlap": len(audit_top20 & prior_set),
                    "features": sorted(audit_top20 & prior_set),
                    "jaccard": round(
                        len(audit_top20 & prior_set) / max(len(audit_top20 | prior_set), 1), 4
                    ),
                }

            summary["audits"][token_selection][source] = audit
            logger.info(
                "%s/%s: train_pairs=%d val_pairs=%d | transfer_top20=%.3f | prior_cross_overlap=%s",
                token_selection,
                source,
                len(train_pairs),
                len(val_pairs),
                next(
                    (row["positive_score_rate"] for row in audit["transfer"] if row["top_k"] == 20),
                    float("nan"),
                ),
                audit.get("overlap_with_prior_crosscoder_top20", {}).get("n_overlap", "N/A"),
            )

    _plot_scatter(summary, output_dir)
    _plot_transfer(summary, output_dir)
    _plot_overlap(summary, output_dir)
    _plot_pair_score_hist(summary, output_dir)

    clean_summary = _strip_arrays(summary)
    (output_dir / "confounder_feature_audit_summary.json").write_text(
        json.dumps(clean_summary, indent=2)
    )
    with (output_dir / "confounder_feature_top_features.jsonl").open("w") as fd:
        for token_selection, by_source in clean_summary["audits"].items():
            for source, audit in by_source.items():
                for rank, row in enumerate(audit["top_features"], start=1):
                    fd.write(
                        json.dumps({
                            "token_selection": token_selection,
                            "source": source,
                            "rank": rank,
                            **row,
                        }) + "\n"
                    )
    with (output_dir / "validation_pair_feature_scores.jsonl").open("w") as fd:
        for token_selection, by_source in clean_summary["audits"].items():
            for source, audit in by_source.items():
                for row in audit.get("validation_pair_scores", []):
                    fd.write(
                        json.dumps({
                            "token_selection": token_selection,
                            "source": source,
                            **row,
                        }) + "\n"
                    )

    logger.info("Wrote crosscoder confounder audit to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(CrosscoderConfounderAuditConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run_crosscoder_audit(cfg)


if __name__ == "__main__":
    main()
