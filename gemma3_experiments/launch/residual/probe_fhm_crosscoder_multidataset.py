

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
import torch
from safetensors import safe_open
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.probe_fhm_confounder_readout import LABELS as FHM_LABELS
from gemma3_experiments.launch.residual.probe_fhm_crosscoder import (
    DenseLayerSpec,
    DenseShardReader,
    GemmaCrosscoder,
    _aggregate_acts,
    _common_keys,
    _manifest_key,
    _role_mask,
    _top_classifier_features,
)
from gemma3_experiments.launch.residual.probe_validate import (
    _build_fhm_metadata_hard_negative_pairs,
    _calibrate_threshold,
    _load_fhm_confounder_metadata,
    _per_class_metrics,
    _read_jsonl,
    _row_gold_label,
    _row_sample_key,
)
from gemma3_experiments.residual_sae import resolve_visible_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Unified binary labels used during training
UNIFIED_HATE = "hate"
UNIFIED_SAFE = "safe"
UNIFIED_LABELS = [UNIFIED_SAFE, UNIFIED_HATE]  # alphabetical → class indices 0/1

# Raw label → unified binary
_HATE_RAW = frozenset({
    "hateful",          # FHM
    "Hate Speech",      # Crisis subtask A
    "Racist", "Sexist", "Homophobe", "OtherHate", "Religion",  # MMHS
})
_SAFE_RAW = frozenset({
    "not hateful",      # FHM
    "No Hate Speech",   # Crisis subtask A
    "NotHate",          # MMHS
})

# Which task_name to include per dataset
_DATASET_TASK_ALLOW = {
    "Hateful_Memes_Dataset": {"hateful_memes_binary"},
    "MMHS150K": {"hate_category"},
    "Crisis_HateMM": {"subtask_a_hate_speech"},
}


def _to_binary(raw_label: str | None) -> str | None:
    if raw_label in _HATE_RAW:
        return UNIFIED_HATE
    if raw_label in _SAFE_RAW:
        return UNIFIED_SAFE
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MultiDatasetCrosscoderConfig:
    # Crosscoder
    crosscoder_repo_id: str = "google/gemma-scope-2-4b-it"
    crosscoder_hookpoint: str = "crosscoder/layer_9_17_22_29_width_65k_l0_medium"

    # Layer names (for logging / shard cache keys)
    layer9_name: str = "layer9"
    layer17_name: str = "layer17"
    layer22_name: str = "layer22"
    layer29_name: str = "layer29"

    # Layer-22: the existing all-dataset dense cache covers FHM+MMHS+Crisis
    layer22_train_dense_cache_dir: str = ""
    layer22_val_dense_cache_dir: str = ""

    # Layers 9/17/29 train: FHM rows from one cache; MMHS+Crisis from another
    # If supplement dirs are empty, only FHM rows are used (FHM-only mode).
    layer9_train_fhm_cache_dir: str = ""
    layer9_train_supplement_cache_dir: str = ""   # MMHS+Crisis
    layer9_val_dense_cache_dir: str = ""

    layer17_train_fhm_cache_dir: str = ""
    layer17_train_supplement_cache_dir: str = ""
    layer17_val_dense_cache_dir: str = ""

    layer29_train_fhm_cache_dir: str = ""
    layer29_train_supplement_cache_dir: str = ""
    layer29_val_dense_cache_dir: str = ""

    # Which datasets to include in training
    train_datasets: list[str] = field(
        default_factory=lambda: ["Hateful_Memes_Dataset", "MMHS150K", "Crisis_HateMM"]
    )

    # Sample weighting: "inverse_freq" (default), "dataset_class" (balanced both axes), "none"
    sample_weight_strategy: str = "inverse_freq"

    # FHM pseudo-HN (applied to FHM rows only)
    fhm_confounder_metadata_path: str = ""
    include_pair_delta_rows: bool = True
    pair_sources: list[str] = field(default_factory=lambda: ["pseudo_img", "pseudo_text"])

    # Output
    baseline_predictions_path: str = ""
    output_dir: str = ""
    eval_splits: list[str] = field(default_factory=lambda: ["validation", "val", "dev"])

    # Feature extraction
    token_selection: str = "all"
    aggregation: str = "max"
    batch_size: int = 64
    chunk_size: int = 64
    shard_cache_size: int = 16

    # Classifier
    classifier_type: str = "mlp"
    mlp_hidden_size: int = 512
    mlp_alpha: float = 1e-4
    mlp_max_iter: int = 200
    C: float = 1.0
    max_iter: int = 5000

    device: str = "cuda:0"


# ---------------------------------------------------------------------------
# Row loading
# ---------------------------------------------------------------------------

def _load_rows_from_cache(
    cache_dir: Path,
    split_names: set[str],
    train_datasets: set[str],
) -> list[dict[str, Any]]:
    """Load rows that match the given splits and datasets, tagging with binary label."""
    rows = []
    for row in _read_jsonl(cache_dir / "manifest.jsonl"):
        ds = row.get("dataset_name", "")
        if ds not in train_datasets:
            continue
        allowed = _DATASET_TASK_ALLOW.get(ds, set())
        if row.get("task_name", "") not in allowed:
            continue
        if row.get("split") not in split_names:
            continue
        raw = _row_gold_label(row)
        binary = _to_binary(raw)
        if binary is None:
            continue
        row = dict(row)
        row["_cache_dir"] = str(cache_dir)
        row["_binary_label"] = binary
        row["_dataset"] = ds
        rows.append(row)
    return rows


def _build_layer_train_rows(
    cfg: MultiDatasetCrosscoderConfig,
) -> tuple[list[list[dict[str, Any]]], list[list[str]]]:
    """Return (rows_by_layer, key_lists) for 4 layers, merging supplement caches."""
    ds_set = set(cfg.train_datasets)

    # Layer-22: single all-dataset cache
    l22_rows = _load_rows_from_cache(
        Path(cfg.layer22_train_dense_cache_dir), {"train"}, ds_set
    )

    def _merge_layer_rows(fhm_dir: str, supp_dir: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if fhm_dir:
            rows += _load_rows_from_cache(Path(fhm_dir), {"train"}, ds_set)
        if supp_dir:
            rows += _load_rows_from_cache(Path(supp_dir), {"train"}, ds_set)
        return rows

    l9_rows  = _merge_layer_rows(cfg.layer9_train_fhm_cache_dir,  cfg.layer9_train_supplement_cache_dir)
    l17_rows = _merge_layer_rows(cfg.layer17_train_fhm_cache_dir, cfg.layer17_train_supplement_cache_dir)
    l29_rows = _merge_layer_rows(cfg.layer29_train_fhm_cache_dir, cfg.layer29_train_supplement_cache_dir)

    rows_by_layer = [l9_rows, l17_rows, l22_rows, l29_rows]
    key_lists = [[_manifest_key(r) for r in layer_rows] for layer_rows in rows_by_layer]
    for name, cnt in zip(
        [cfg.layer9_name, cfg.layer17_name, cfg.layer22_name, cfg.layer29_name],
        [len(r) for r in rows_by_layer],
    ):
        logger.info("  Train layer %s: %d rows", name, cnt)
    return rows_by_layer, key_lists


def _build_layer_val_rows(
    cfg: MultiDatasetCrosscoderConfig,
) -> tuple[list[list[dict[str, Any]]], list[list[str]]]:
    """Return (rows_by_layer, key_lists) for FHM validation, all 4 layers."""
    eval_set = set(cfg.eval_splits)
    fhm_only = {"Hateful_Memes_Dataset"}

    l22_rows = _load_rows_from_cache(
        Path(cfg.layer22_val_dense_cache_dir), eval_set, fhm_only
    )

    def _val_layer(cache_dir: str) -> list[dict[str, Any]]:
        if not cache_dir:
            return []
        return _load_rows_from_cache(Path(cache_dir), eval_set, fhm_only)

    l9_rows  = _val_layer(cfg.layer9_val_dense_cache_dir)
    l17_rows = _val_layer(cfg.layer17_val_dense_cache_dir)
    l29_rows = _val_layer(cfg.layer29_val_dense_cache_dir)

    rows_by_layer = [l9_rows, l17_rows, l22_rows, l29_rows]
    key_lists = [[_manifest_key(r) for r in layer_rows] for layer_rows in rows_by_layer]
    for name, cnt in zip(
        [cfg.layer9_name, cfg.layer17_name, cfg.layer22_name, cfg.layer29_name],
        [len(r) for r in rows_by_layer],
    ):
        logger.info("  Val layer %s: %d rows", name, cnt)
    return rows_by_layer, key_lists


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def _compute_sample_weights(
    rows: list[dict[str, Any]],
    strategy: str,
) -> np.ndarray | None:
    """Compute per-sample weights so each (dataset) or (dataset, class) cell is balanced."""
    if strategy == "none":
        return None
    if strategy == "inverse_freq":
        ds_counts = Counter(r["_dataset"] for r in rows)
        n_ds = len(ds_counts)
        n_total = len(rows)
        return np.array(
            [n_total / (n_ds * ds_counts[r["_dataset"]]) for r in rows], dtype=np.float32
        )
    if strategy == "dataset_class":
        cell_counts = Counter((r["_dataset"], r["_binary_label"]) for r in rows)
        n_cells = len(cell_counts)
        n_total = len(rows)
        return np.array(
            [n_total / (n_cells * cell_counts[(r["_dataset"], r["_binary_label"])]) for r in rows],
            dtype=np.float32,
        )
    raise ValueError(f"Unknown sample_weight_strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Feature extraction (reuses DenseShardReader from probe_fhm_crosscoder)
# ---------------------------------------------------------------------------

def _extract_features(
    rows_by_layer: list[list[dict[str, Any]]],
    key_lists: list[list[str]],
    layer_specs: list[DenseLayerSpec],
    crosscoder: GemmaCrosscoder,
    cfg: MultiDatasetCrosscoderConfig,
) -> tuple[list[str], Any]:
    common_keys = _common_keys(key_lists)
    row_lookup = [{_manifest_key(r): r for r in rows} for rows in rows_by_layer]
    reader = DenseShardReader(layer_specs, cfg.shard_cache_size, cfg.token_selection)
    feature_rows = []
    for batch_start in range(0, len(common_keys), cfg.batch_size):
        batch = common_keys[batch_start : batch_start + cfg.batch_size]
        for key in batch:
            sample_rows = [lookup[key] for lookup in row_lookup]
            hidden_tld, roles = reader.get_sample(sample_rows)
            mask = _role_mask(roles, cfg.token_selection)
            selected = hidden_tld[mask]
            chunks = []
            for s in range(0, selected.shape[0], cfg.chunk_size):
                chunks.append(crosscoder.encode(selected[s : s + cfg.chunk_size]))
            if chunks:
                acts = torch.cat(chunks, dim=0)
                vec = _aggregate_acts(acts, cfg.aggregation)
            else:
                vec = np.zeros((crosscoder.num_latents,), dtype=np.float32)
            feature_rows.append(scipy.sparse.csr_matrix(vec.reshape(1, -1)))
        if (batch_start + cfg.batch_size) % 512 == 0 or batch_start + cfg.batch_size >= len(common_keys):
            logger.info(
                "  Features: %d / %d",
                min(batch_start + cfg.batch_size, len(common_keys)),
                len(common_keys),
            )
    return common_keys, scipy.sparse.vstack(feature_rows, format="csr", dtype=np.float32)


# ---------------------------------------------------------------------------
# Pair metrics (local version that accepts positive_class)
# ---------------------------------------------------------------------------

def _pair_metrics_v2(
    pairs: list[tuple[str, str]],
    pred_by_key: dict[str, str],
    proba_by_key: dict[str, dict[str, float]],
    positive_class: str,
    negative_class: str,
) -> dict[str, Any]:
    rows_out = []
    for hateful_key, benign_key in pairs:
        if hateful_key not in pred_by_key or benign_key not in pred_by_key:
            continue
        hateful_pred = pred_by_key[hateful_key]
        benign_pred = pred_by_key[benign_key]
        hateful_prob = proba_by_key[hateful_key].get(positive_class, 0.0)
        benign_prob = proba_by_key[benign_key].get(positive_class, 0.0)
        rows_out.append({
            "hateful_key": hateful_key,
            "benign_key": benign_key,
            "both_correct": hateful_pred == positive_class and benign_pred == negative_class,
            "separated": hateful_pred != benign_pred,
            "positive_margin": hateful_prob > benign_prob,
            "prob_gap": hateful_prob - benign_prob,
        })
    if not rows_out:
        return {"n_pairs": 0, "rows": []}
    return {
        "n_pairs": len(rows_out),
        "both_correct_rate": round(float(np.mean([r["both_correct"] for r in rows_out])), 6),
        "separated_rate": round(float(np.mean([r["separated"] for r in rows_out])), 6),
        "positive_margin_rate": round(float(np.mean([r["positive_margin"] for r in rows_out])), 6),
        "median_prob_gap": round(float(np.median([r["prob_gap"] for r in rows_out])), 6),
        "rows": rows_out,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_multidataset_probe(cfg: MultiDatasetCrosscoderConfig) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(cfg.baseline_predictions_path) if cfg.baseline_predictions_path else None
    metadata_path = Path(cfg.fhm_confounder_metadata_path) if cfg.fhm_confounder_metadata_path else None
    if metadata_path:
        assert metadata_path.exists(), f"FHM metadata missing: {metadata_path}"

    # --- Layer specs for DenseShardReader ---
    layer_specs = [
        DenseLayerSpec(cfg.layer9_name, "", ""),
        DenseLayerSpec(cfg.layer17_name, "", ""),
        DenseLayerSpec(cfg.layer22_name, "", ""),
        DenseLayerSpec(cfg.layer29_name, "", ""),
    ]

    # --- Load train rows ---
    logger.info("Loading train rows...")
    train_rows_by_layer, train_key_lists = _build_layer_train_rows(cfg)
    common_train_keys = _common_keys(train_key_lists)
    logger.info("Common train keys across 4 layers: %d", len(common_train_keys))

    # Build a representative label/dataset lookup from any layer (all share same metadata)
    train_row_by_key: dict[str, dict[str, Any]] = {}
    for rows in train_rows_by_layer:
        for r in rows:
            k = _manifest_key(r)
            if k not in train_row_by_key:
                train_row_by_key[k] = r

    y_train_full = [train_row_by_key[k]["_binary_label"] for k in common_train_keys]
    ds_train_full = [train_row_by_key[k]["_dataset"] for k in common_train_keys]

    # Log per-dataset stats
    ds_label_counts: Counter = Counter(zip(ds_train_full, y_train_full))
    for (ds, lbl), cnt in sorted(ds_label_counts.items()):
        logger.info("  train %s / %s: %d", ds, lbl, cnt)

    # --- Load val rows (FHM only) ---
    logger.info("Loading FHM validation rows...")
    val_rows_by_layer, val_key_lists = _build_layer_val_rows(cfg)
    common_val_keys = _common_keys(val_key_lists)
    logger.info("Common FHM val keys: %d", len(common_val_keys))
    val_row_by_key = {_manifest_key(r): r for r in val_rows_by_layer[2]}  # layer-22 reference
    y_val = [val_row_by_key[k]["_binary_label"] for k in common_val_keys]

    # --- Load crosscoder ---
    first_train_row = train_rows_by_layer[2][0]  # layer-22 row
    first_shard = Path(first_train_row["_cache_dir"]) / first_train_row["shard_file"]
    with safe_open(str(first_shard), framework="pt", device="cpu") as fd:
        d_model = int(fd.get_tensor("hidden").shape[1])
    device = resolve_visible_device(cfg.device)
    logger.info("Loading crosscoder %s (d_model=%d)", cfg.crosscoder_hookpoint, d_model)
    crosscoder = GemmaCrosscoder.load_from_hub(
        cfg.crosscoder_repo_id,
        cfg.crosscoder_hookpoint,
        n_layers=len(layer_specs),
        d_model=d_model,
        device=device,
    )
    logger.info("Crosscoder num_latents=%d", crosscoder.num_latents)

    # --- Feature extraction ---
    logger.info("Extracting train features...")
    train_keys, X_train_base = _extract_features(
        train_rows_by_layer, train_key_lists, layer_specs, crosscoder, cfg
    )
    logger.info("Extracting val features...")
    val_keys, X_val = _extract_features(
        val_rows_by_layer, val_key_lists, layer_specs, crosscoder, cfg
    )

    y_train = [train_row_by_key[k]["_binary_label"] for k in train_keys]
    ds_train = [train_row_by_key[k]["_dataset"] for k in train_keys]

    # --- FHM pseudo-HN delta rows ---
    X_train = X_train_base
    hard_negative_stats = None
    if cfg.include_pair_delta_rows and metadata_path:
        fhm_rows = [r for r in train_rows_by_layer[2] if r.get("_dataset") == "Hateful_Memes_Dataset"]
        fhm_key_to_idx = {_manifest_key(r): i for i, r in enumerate(train_rows_by_layer[2]) if
                          _manifest_key(r) in set(train_keys)}
        # Restrict X indices to the actual extracted keys
        train_key_to_idx = {k: i for i, k in enumerate(train_keys)}
        delta_csr, delta_labels, hard_negative_stats = _build_fhm_metadata_hard_negative_pairs(
            fhm_rows, X_train_base, train_key_to_idx, metadata_path, cfg.pair_sources
        )
        # _build_fhm_metadata_hard_negative_pairs returns raw FHM labels ("hateful" /
        # "not hateful"); remap them to the unified binary labels used by y_train.
        delta_labels = [_to_binary(lbl) or lbl for lbl in delta_labels]
        if delta_csr.shape[0]:
            X_train = scipy.sparse.vstack([X_train_base, delta_csr], format="csr")
            y_train = list(y_train) + delta_labels
            ds_train = list(ds_train) + ["Hateful_Memes_Dataset"] * len(delta_labels)
            logger.info("Pair delta rows: %d -> %d", len(train_keys), X_train.shape[0])

    # --- Sample weights ---
    full_rows_for_weights = [
        {"_dataset": ds, "_binary_label": lbl}
        for ds, lbl in zip(ds_train, y_train)
    ]
    sample_weights = _compute_sample_weights(full_rows_for_weights, cfg.sample_weight_strategy)
    if sample_weights is not None:
        logger.info(
            "Sample weights: mean=%.4f std=%.4f min=%.4f max=%.4f",
            sample_weights.mean(), sample_weights.std(), sample_weights.min(), sample_weights.max(),
        )

    # --- Scale + fit ---
    scaler = MaxAbsScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    classes = le.classes_.tolist()
    assert set(classes) == {UNIFIED_HATE, UNIFIED_SAFE}, f"Unexpected classes: {classes}"

    logger.info("Training %s classifier...", cfg.classifier_type)
    if cfg.classifier_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(cfg.mlp_hidden_size,),
            activation="relu",
            alpha=cfg.mlp_alpha,
            max_iter=cfg.mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )
        sw = sample_weights if sample_weights is not None else None
        clf.fit(X_train_sc.toarray(), y_train_enc, sample_weight=sw)
        y_pred_enc = clf.predict(X_val_sc.toarray())
        y_proba = clf.predict_proba(X_val_sc.toarray())
    else:
        clf = LogisticRegression(
            C=cfg.C, max_iter=cfg.max_iter, solver="liblinear"
        )
        clf.fit(X_train_sc, y_train_enc, sample_weight=sample_weights)
        y_pred_enc = clf.predict(X_val_sc)
        y_proba = clf.predict_proba(X_val_sc)

    y_pred_unified = le.inverse_transform(y_pred_enc).tolist()

    # --- Remap unified labels → FHM labels for reporting ---
    _remap = {UNIFIED_HATE: "hateful", UNIFIED_SAFE: "not hateful"}
    y_pred = [_remap[p] for p in y_pred_unified]
    y_val_fhm = [_remap[v] for v in y_val]
    proba_rows = [
        {"hateful": float(y_proba[i, classes.index(UNIFIED_HATE)]),
         "not hateful": float(y_proba[i, classes.index(UNIFIED_SAFE)])}
        for i in range(len(val_keys))
    ]

    # --- Metrics ---
    raw_metrics = _per_class_metrics(y_val_fhm, y_pred)
    best_t, _, best_cal_preds = _calibrate_threshold(y_val_fhm, y_proba, ["not hateful", "hateful"], "hateful")
    calibrated_metrics = _per_class_metrics(y_val_fhm, best_cal_preds)
    calibrated_metrics["threshold"] = round(best_t, 2)

    threshold_sweep = []
    pos_idx = classes.index(UNIFIED_HATE)
    for t_int in range(10, 91):
        t = t_int / 100.0
        preds_t = ["hateful" if p[pos_idx] >= t else "not hateful" for p in y_proba]
        m = _per_class_metrics(y_val_fhm, preds_t)
        threshold_sweep.append({"threshold": t, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})

    # --- Baseline ---
    baseline_metrics = None
    baseline_by_key: dict[str, Any] = {}
    if baseline_path and baseline_path.exists():
        for row in _read_jsonl(baseline_path):
            baseline_by_key[f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"] = row
        b_gold, b_pred = [], []
        for k, g in zip(val_keys, y_val_fhm):
            bp = baseline_by_key.get(k, {}).get("prediction_label")
            if bp in {"hateful", "not hateful"}:
                b_gold.append(g)
                b_pred.append(bp)
        if b_gold:
            baseline_metrics = _per_class_metrics(b_gold, b_pred)

    # --- Pair metrics (FHM confounders) ---
    pred_by_key = dict(zip(val_keys, y_pred))
    proba_by_key = dict(zip(val_keys, proba_rows))
    pair_metrics: dict[str, Any] = {}
    if metadata_path:
        from gemma3_experiments.launch.residual.analyze_fhm_confounder_features import _build_pairs
        metadata = _load_fhm_confounder_metadata(metadata_path)
        fhm_val_rows = [val_row_by_key[k] for k in val_keys if k in val_row_by_key]
        for source in cfg.pair_sources:
            pairs = _build_pairs(fhm_val_rows, metadata, set(val_keys), source)
            pair_metrics[source] = _pair_metrics_v2(
                pairs, pred_by_key, proba_by_key, "hateful", "not hateful"
            )

    top_features = _top_classifier_features(clf, classes)
    # Remap class keys for top_features output
    top_features = {
        _remap.get(k, k): v for k, v in top_features.items()
    }

    # --- Write outputs ---
    ds_summary = {}
    for (ds, lbl), cnt in Counter(zip(ds_train, y_train)).items():
        ds_summary.setdefault(ds, {})[lbl] = cnt

    metrics = {
        "config": vars(cfg),
        "crosscoder_hookpoint": cfg.crosscoder_hookpoint,
        "n_features_total": crosscoder.num_latents,
        "train_dataset_summary": ds_summary,
        "n_train_real": len(train_keys),
        "n_train_after_delta": int(X_train.shape[0]),
        "n_val": len(val_keys),
        "hard_negative_stats": hard_negative_stats,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "baseline_metrics": baseline_metrics,
        "threshold_sweep": threshold_sweep,
        "pair_metrics": {
            src: {k: v for k, v in pm.items() if k != "rows"}
            for src, pm in pair_metrics.items()
        },
        "top_classifier_features": top_features,
    }
    (output_dir / "multidataset_probe_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "crosscoder_top_features.json").write_text(json.dumps(top_features, indent=2))

    with (output_dir / "probe_val_predictions.jsonl").open("w") as fd:
        for key, gold, pred, proba in zip(val_keys, y_val_fhm, y_pred, proba_rows):
            base_row = baseline_by_key.get(key)
            fd.write(json.dumps({
                "sample_key": key,
                "dataset_name": "Hateful_Memes_Dataset",
                "task_name": "hateful_memes_binary",
                "sample_id": key.split("::")[-1],
                "gold_label": gold,
                "probe_pred": pred,
                "probe_proba": proba,
                "baseline_pred": base_row.get("prediction_label") if base_row else None,
                "probe_correct": pred == gold,
                "baseline_correct": (base_row.get("prediction_label") == gold) if base_row else None,
            }) + "\n")

    if pair_metrics:
        with (output_dir / "validation_pair_metrics.jsonl").open("w") as fd:
            for source, pm in pair_metrics.items():
                for row in pm.get("rows", []):
                    fd.write(json.dumps({"source": source, **row}) + "\n")

    logger.info(
        "Multi-dataset probe: raw_f1=%.4f cal_f1=%.4f baseline_f1=%.4f n_features=%d",
        raw_metrics["macro_f1"],
        calibrated_metrics["macro_f1"],
        baseline_metrics["macro_f1"] if baseline_metrics else float("nan"),
        crosscoder.num_latents,
    )
    logger.info("Wrote outputs to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(MultiDatasetCrosscoderConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run_multidataset_probe(cfg)


if __name__ == "__main__":
    main()
