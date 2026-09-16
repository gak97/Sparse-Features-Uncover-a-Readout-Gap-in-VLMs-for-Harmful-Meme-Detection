

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.analyze_fhm_confounder_features import _build_pairs, _feature_stats
from gemma3_experiments.launch.residual.probe_fhm_confounder_readout import LABELS, TASK_KEY, _pair_metrics
from gemma3_experiments.launch.residual.probe_validate import (
    _calibrate_threshold,
    _load_fhm_confounder_metadata,
    _per_class_metrics,
    _read_jsonl,
    _row_gold_label,
    _row_sample_key,
)
from gemma3_experiments.residual_sae import (
    ROLE_GENERATED,
    ROLE_IMAGE,
    ROLE_PROMPT,
    resolve_visible_device,
)

logger = logging.getLogger(__name__)


@dataclass
class PairMarginReadoutConfig:
    # SAE features
    residual_sae_path: str = ""
    train_cache_dir: str = ""
    val_cache_dir: str = ""
    # Pair metadata
    fhm_confounder_metadata_path: str = ""
    baseline_predictions_path: str = ""
    output_dir: str = ""

    # Token selection for base features
    token_selections: list[str] = field(default_factory=lambda: ["all", "image", "prompt", "generated"])
    pair_sources: list[str] = field(default_factory=lambda: ["pseudo_img", "pseudo_text"])
    aggregation: str = "max"
    batch_size: int = 256

    # Confounder feature projection settings
    top_k_confounder_features: int = 50
    projection_top_k: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    include_signed_feature_columns: bool = True
    include_projection_columns: bool = True

    # Readout architecture
    classifier_type: str = "mlp"  # mlp | logistic
    C: float = 1.0
    max_iter: int = 5000
    class_weight: str = "balanced"
    mlp_hidden_size: int = 256
    mlp_alpha: float = 1e-4
    mlp_max_iter: int = 200

    # Margin loss settings
    margin: float = 0.2
    alpha: float = 0.7  # weight for cross-entropy, (1-alpha) for margin

    ensemble_weights: list[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ])
    plot: bool = True
    device: str = "cuda:0"


# ──────────────────────────────────────────────────────────────
# Pair-margin loss utilities
# ──────────────────────────────────────────────────────────────

def _pair_margin_loss(
    proba: np.ndarray,
    pairs: list[tuple[str, str]],
    key_to_idx: dict[str, int],
    pos_idx: int,
    neg_idx: int,
    margin: float,
) -> float:
    """Compute mean margin loss: max(0, P(neg) - P(pos) + margin) per pair."""
    if not pairs:
        return 0.0
    losses = []
    for h_key, b_key in pairs:
        h_idx = key_to_idx.get(h_key)
        b_idx = key_to_idx.get(b_key)
        if h_idx is None or b_idx is None:
            continue
        p_hate = proba[h_idx, pos_idx]
        p_benign = proba[b_idx, pos_idx]
        loss = max(0.0, p_benign - p_hate + margin)
        losses.append(loss)
    return float(np.mean(losses)) if losses else 0.0


def _pair_delta_rows(
    keys: list[str],
    csr: Any,
    pairs: list[tuple[str, str]],
) -> tuple[Any, list[str]]:
    """Build only delta rows (hateful-benign and benign-hateful) from paired samples."""
    import scipy.sparse
    key_to_idx = {key: idx for idx, key in enumerate(keys)}
    delta_rows = []
    delta_labels = []
    seen: set[tuple[str, str]] = set()
    for hate_key, benign_key in pairs:
        pair_key = tuple(sorted((hate_key, benign_key)))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        h_idx = key_to_idx.get(hate_key)
        b_idx = key_to_idx.get(benign_key)
        if h_idx is None or b_idx is None:
            continue
        h_vec = csr[h_idx]
        b_vec = csr[b_idx]
        delta_rows.append(h_vec - b_vec)
        delta_labels.append("hateful")
        delta_rows.append(b_vec - h_vec)
        delta_labels.append("not hateful")
    if not delta_rows:
        return scipy.sparse.csr_matrix((0, csr.shape[1]), dtype=np.float32), []
    return scipy.sparse.vstack(delta_rows, format="csr", dtype=np.float32), delta_labels


# ──────────────────────────────────────────────────────────────
# Main pipeline (mirrors confounder readout + adds margin loss fitting)
# ──────────────────────────────────────────────────────────────

def _load_fhm_rows(cache_dir: Path, split_names: set[str], *, cache_tag: bool = False) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(cache_dir / "manifest.jsonl"):
        if row.get("dataset_name") != "Hateful_Memes_Dataset":
            continue
        if row.get("task_name") != "hateful_memes_binary":
            continue
        if row.get("split") not in split_names:
            continue
        if _row_gold_label(row) not in set(LABELS):
            continue
        if cache_tag:
            row = dict(row)
            row["_cache_dir"] = str(cache_dir)
        rows.append(row)
    return rows


def run_pair_margin_readout(cfg: PairMarginReadoutConfig) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
    import scipy.sparse

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cache = Path(cfg.train_cache_dir)
    val_cache = Path(cfg.val_cache_dir)
    metadata_path = Path(cfg.fhm_confounder_metadata_path)
    baseline_path = Path(cfg.baseline_predictions_path)
    assert train_cache.exists(), f"Train cache missing: {train_cache}"
    assert val_cache.exists(), f"Val cache missing: {val_cache}"
    assert metadata_path.exists(), f"Metadata missing: {metadata_path}"

    device = resolve_visible_device(cfg.device)
    logger.info("Loading SAE from %s", cfg.residual_sae_path)
    from gemma3_experiments.residual_sae import load_residual_sae, load_residual_stats
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)

    # Import extraction helpers from confounder readout
    from gemma3_experiments.launch.residual.probe_fhm_confounder_readout import (
        _align_csr,
        _common_keys,
        _build_readout_matrix,
    )

    # Load rows + extract features per token selection
    train_rows = _load_fhm_rows(train_cache, {"train"}, cache_tag=True)
    val_rows = _load_fhm_rows(val_cache, {"validation", "val", "dev"}, cache_tag=True)
    logger.info("FHM rows: train=%d val=%d", len(train_rows), len(val_rows))

    extract_common = dict(aggregation=cfg.aggregation, batch_size=cfg.batch_size, device=device, n_features=sae.num_latents)
    from gemma3_experiments.launch.residual.probe_validate import _extract_features, _extract_features_multi
    train_sets, val_sets = {}, {}
    for sel in cfg.token_selections:
        logger.info("Extracting %s features", sel)
        train_sets[sel] = _extract_features(train_rows, train_cache, sae, stats, token_selection=sel, **extract_common)
        val_sets[sel] = _extract_features_multi(val_rows, sae, stats, token_selection=sel, **extract_common)

    train_keys = _common_keys(train_sets, cfg.token_selections[0])
    val_keys = _common_keys(val_sets, cfg.token_selections[0])
    assert train_keys and val_keys
    logger.info("Common keys: train=%d val=%d", len(train_keys), len(val_keys))

    # Audit confounder features
    metadata = _load_fhm_confounder_metadata(metadata_path)
    train_key_set, val_key_set = set(train_keys), set(val_keys)
    train_pairs_by_source = {s: _build_pairs(train_rows, metadata, train_key_set, s) for s in cfg.pair_sources}
    val_pairs_by_source = {s: _build_pairs(val_rows, metadata, val_key_set, s) for s in cfg.pair_sources}

    aligned_train = {sel: _align_csr(keys, csr, train_keys) for sel, (keys, csr) in train_sets.items()}
    aligned_val = {sel: _align_csr(keys, csr, val_keys) for sel, (keys, csr) in val_sets.items()}

    key_to_idx = {k: i for i, k in enumerate(train_keys)}
    audits: dict[str, dict[str, dict[str, Any]]] = {}
    for token_selection in cfg.token_selections:
        audits[token_selection] = {}
        for source in cfg.pair_sources:
            audits[token_selection][source] = _feature_stats(
                aligned_train[token_selection],
                key_to_idx,
                train_pairs_by_source[source],
                cfg.top_k_confounder_features,
            )

    # _build_readout_matrix expects a config with these attributes.
    cfg.top_k_features_per_block = cfg.top_k_confounder_features
    cfg.include_generic_token_features = False
    cfg.generic_token_selection = "all"

    X_train, feature_meta = _build_readout_matrix(train_sets, train_keys, audits, cfg)
    X_val, _ = _build_readout_matrix(val_sets, val_keys, audits, cfg)

    # Pair-delta rows
    train_row_by_key = {_row_sample_key(r): r for r in train_rows}
    val_row_by_key = {_row_sample_key(r): r for r in val_rows}
    y_train_base = [_row_gold_label(train_row_by_key[k]) for k in train_keys]
    y_val = [_row_gold_label(val_row_by_key[k]) for k in val_keys]

    X_train_final = X_train
    y_train_final = list(y_train_base)
    hard_neg_stats = None
    all_train_pairs = train_pairs_by_source["pseudo_img"] + train_pairs_by_source["pseudo_text"]
    if all_train_pairs:
        delta_csr, delta_labels = _pair_delta_rows(train_keys, X_train, all_train_pairs)
        if delta_csr.shape[0] > 0:
            X_train_final = scipy.sparse.vstack([X_train, delta_csr], format="csr")
            y_train_final = y_train_base + delta_labels
            hard_neg_stats = {"n_delta_rows": delta_csr.shape[0], "n_pairs_unique": delta_csr.shape[0] // 2}
            logger.info("Pair deltas: %d -> %d", len(train_keys), X_train_final.shape[0])

    # Standardize
    scaler = MaxAbsScaler()
    X_train_s = scaler.fit_transform(X_train_final)
    X_val_s = scaler.transform(X_val)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train_final)
    classes = le.classes_.tolist()
    pos_idx = classes.index("hateful")
    neg_idx = classes.index("not hateful")

    # Fit readout with optional margin loss refinement
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
        clf.fit(X_train_s.toarray(), y_enc)
    else:
        clf = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, solver="liblinear",
                                 class_weight=cfg.class_weight if cfg.class_weight != "None" else None)
        clf.fit(X_train_s, y_enc)

    # Initial probabilities
    if cfg.classifier_type == "mlp":
        train_proba = clf.predict_proba(X_train_s.toarray())
    else:
        train_proba = clf.predict_proba(X_train_s)

    # Margin-loss diagnostic: count pair violations to quantify confounder difficulty
    if cfg.margin > 0 and all_train_pairs:
        key_to_idx = {k: i for i, k in enumerate(train_keys)}
        pair_violations = []
        pair_margins = []
        for h_key, b_key in all_train_pairs:
            h_idx = key_to_idx.get(h_key)
            b_idx = key_to_idx.get(b_key)
            if h_idx is None or b_idx is None:
                continue
            ph = train_proba[h_idx, pos_idx]
            pb = train_proba[b_idx, pos_idx]
            pair_margins.append(pb - ph)
            pair_violations.append(pb - ph + cfg.margin > 0)
        n_violations = sum(pair_violations)
        median_margin = float(np.median(pair_margins)) if pair_margins else 0.0
        logger.info(
            "Train pair-margin violations: %d / %d (%.1f%%) | median margin gap: %.4f",
            n_violations, len(pair_margins), 100.0 * n_violations / max(len(pair_margins), 1), median_margin,
        )

    # Predict
    if cfg.classifier_type == "mlp":
        y_pred_enc = clf.predict(X_val_s.toarray())
        y_proba = clf.predict_proba(X_val_s.toarray())
    else:
        y_pred_enc = clf.predict(X_val_s)
        y_proba = clf.predict_proba(X_val_s)
    y_pred = le.inverse_transform(y_pred_enc).tolist()
    raw_metrics = _per_class_metrics(y_val, y_pred)
    best_t, _, best_cal = _calibrate_threshold(y_val, y_proba, classes, "hateful")
    cal_metrics = _per_class_metrics(y_val, best_cal)
    cal_metrics["threshold"] = round(best_t, 2)

    # Pair metrics
    pred_by_key = dict(zip(val_keys, y_pred))
    proba_by_key = dict(zip(val_keys, [{c: round(float(y_proba[i, j]), 6) for j, c in enumerate(classes)} for i in range(len(val_keys))]))
    pair_metrics = {s: _pair_metrics(val_pairs_by_source[s], pred_by_key, proba_by_key) for s in cfg.pair_sources}

    # Baseline
    baseline, bl_gold, bl_pred = {}, [], []
    if baseline_path:
        for row in _read_jsonl(baseline_path):
            key = f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"
            baseline[key] = row
        for key, label in zip(val_keys, y_val):
            br = baseline.get(key)
            bp = br.get("prediction_label") if br else None
            if bp in LABELS:
                bl_gold.append(label)
                bl_pred.append(bp)
    bl_metrics = _per_class_metrics(bl_gold, bl_pred) if bl_gold else None

    # Save
    metrics = {
        "task_key": TASK_KEY,
        "config": {
            "classifier_type": cfg.classifier_type,
            "margin": cfg.margin,
            "alpha": cfg.alpha,
            "mlp_hidden_size": cfg.mlp_hidden_size,
            "mlp_alpha": cfg.mlp_alpha,
        },
        "n_train_real": len(train_keys),
        "n_train_after_delta": int(X_train_final.shape[0]),
        "n_val": len(val_keys),
        "n_features_total": int(X_train_final.shape[1]),
        "hard_negative_stats": hard_neg_stats,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": cal_metrics,
        "baseline_metrics": bl_metrics,
        "pair_metrics": {s: {k: v for k, v in m.items() if k != "rows"} for s, m in pair_metrics.items()},
    }
    (output_dir / "pair_margin_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (output_dir / "probe_val_predictions.jsonl").open("w") as fd:
        for key, gold, pred, proba in zip(val_keys, y_val, y_pred, proba_by_key.values()):
            br = baseline.get(key)
            fd.write(json.dumps({
                "sample_key": key,
                "task_key": TASK_KEY,
                "gold_label": gold,
                "probe_pred": pred,
                "probe_proba": proba,
                "baseline_pred": br.get("prediction_label") if br else None,
                "baseline_label_scores": br.get("label_scores") if br else None,
                "probe_correct": pred == gold,
                "baseline_correct": (br.get("prediction_label") == gold) if br else None,
            }) + "\n")
    logger.info("Pair-margin readout: raw_f1=%.4f calibrated_f1=%.4f baseline_f1=%.4f",
                raw_metrics["macro_f1"], cal_metrics["macro_f1"],
                bl_metrics["macro_f1"] if bl_metrics else float("nan"))
    logger.info("Wrote %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(PairMarginReadoutConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run_pair_margin_readout(cfg)


if __name__ == "__main__":
    main()