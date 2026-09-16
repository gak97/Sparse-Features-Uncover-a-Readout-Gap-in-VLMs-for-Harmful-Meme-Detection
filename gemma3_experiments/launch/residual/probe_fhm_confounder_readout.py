

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.analyze_fhm_confounder_features import (
    _build_pairs,
    _feature_stats,
)
from gemma3_experiments.launch.residual.ensemble_probe_baseline import _softmax
from gemma3_experiments.launch.residual.probe_validate import (
    _calibrate_threshold,
    _extract_features,
    _extract_features_multi,
    _load_fhm_confounder_metadata,
    _per_class_metrics,
    _read_jsonl,
    _row_gold_label,
    _row_sample_key,
)
from gemma3_experiments.residual_sae import (
    load_residual_sae,
    load_residual_stats,
    resolve_visible_device,
)

logger = logging.getLogger(__name__)


@dataclass
class FHMConfounderReadoutConfig:
    residual_sae_path: str = ""
    train_cache_dir: str = ""
    val_cache_dir: str = ""
    baseline_predictions_path: str = ""
    output_dir: str = ""
    fhm_confounder_metadata_path: str = ""

    token_selections: list[str] = field(default_factory=lambda: ["all", "image", "prompt", "generated"])
    pair_sources: list[str] = field(default_factory=lambda: ["pseudo_img", "pseudo_text"])
    aggregation: str = "max"
    batch_size: int = 256

    top_k_features_per_block: int = 50
    projection_top_k: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    include_signed_feature_columns: bool = True
    include_projection_columns: bool = True
    # Hybrid mode: append the original SAE feature vector for one token selection.
    # The compact confounder features carry pair-aware signal, but the previous run
    # showed they lose some generic hate/not-hate signal. Appending all-token SAE
    # features gives the readout both broad harm evidence and confounder evidence.
    include_generic_token_features: bool = False
    generic_token_selection: str = "all"
    include_pair_delta_rows: bool = True

    classifier_type: str = "logistic"  # logistic | mlp
    C: float = 1.0
    max_iter: int = 5000
    class_weight: str = "balanced"
    mlp_hidden_size: int = 128
    mlp_alpha: float = 1e-3
    mlp_max_iter: int = 300

    ensemble_weights: list[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ])
    plot: bool = True
    device: str = "cuda:0"


TASK_KEY = "Hateful_Memes_Dataset::hateful_memes_binary::label"
LABELS = ["hateful", "not hateful"]


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


def _align_csr(keys: list[str], csr: Any, common_keys: list[str]) -> Any:
    key_to_idx = {key: idx for idx, key in enumerate(keys)}
    return csr[[key_to_idx[key] for key in common_keys]]


def _common_keys(feature_sets: dict[str, tuple[list[str], Any]], reference: str) -> list[str]:
    required = list(feature_sets)
    common = set(feature_sets[required[0]][0])
    for token_selection in required[1:]:
        common &= set(feature_sets[token_selection][0])
    return [key for key in feature_sets[reference][0] if key in common]


def _signed_feature_block(csr: Any, top_features: list[dict[str, Any]], top_k: int) -> tuple[Any, list[dict[str, Any]], list[float]]:
    feature_rows = top_features[:top_k]
    feature_i = np.array([row["feature_idx"] for row in feature_rows], dtype=np.int32)
    signs = [1.0 if row["direction"] == "hateful_up" else -1.0 for row in feature_rows]
    block = csr[:, feature_i].multiply(np.array(signs, dtype=np.float32))
    return block.tocsr(), feature_rows, signs


def _build_readout_matrix(
    feature_sets: dict[str, tuple[list[str], Any]],
    common_keys: list[str],
    audits: dict[str, dict[str, Any]],
    cfg: FHMConfounderReadoutConfig,
) -> tuple[Any, list[dict[str, Any]]]:
    import scipy.sparse

    blocks = []
    feature_meta: list[dict[str, Any]] = []
    projection_cols = []
    projection_meta: list[dict[str, Any]] = []

    aligned = {
        token_selection: _align_csr(keys, csr, common_keys)
        for token_selection, (keys, csr) in feature_sets.items()
    }

    for token_selection in cfg.token_selections:
        for source in cfg.pair_sources:
            audit = audits[token_selection][source]
            signed_block, feature_rows, signs = _signed_feature_block(
                aligned[token_selection], audit["top_features"], cfg.top_k_features_per_block,
            )
            if cfg.include_signed_feature_columns:
                blocks.append(signed_block)
                for local_i, (row, sign) in enumerate(zip(feature_rows, signs)):
                    feature_meta.append({
                        "kind": "signed_feature",
                        "token_selection": token_selection,
                        "source": source,
                        "local_idx": local_i,
                        "feature_idx": row["feature_idx"],
                        "sign": sign,
                        "direction": row["direction"],
                        "train_sign_consistency": row["sign_consistency"],
                        "train_effect": row["effect"],
                    })

            if cfg.include_projection_columns:
                for top_k in cfg.projection_top_k:
                    top_k = min(top_k, signed_block.shape[1])
                    if top_k <= 0:
                        continue
                    col = np.asarray(signed_block[:, :top_k].sum(axis=1)).ravel() / math.sqrt(top_k)
                    projection_cols.append(col.astype(np.float32))
                    projection_meta.append({
                        "kind": "projection",
                        "token_selection": token_selection,
                        "source": source,
                        "top_k": top_k,
                    })

    if projection_cols:
        blocks.append(scipy.sparse.csr_matrix(np.column_stack(projection_cols), dtype=np.float32))
        feature_meta.extend(projection_meta)

    if cfg.include_generic_token_features:
        assert cfg.generic_token_selection in aligned, (
            f"generic_token_selection={cfg.generic_token_selection!r} must be included in token_selections"
        )
        generic = aligned[cfg.generic_token_selection].tocsr().astype(np.float32)
        blocks.append(generic)
        feature_meta.extend(
            {
                "kind": "generic_token_feature",
                "token_selection": cfg.generic_token_selection,
                "feature_idx": int(feature_i),
            }
            for feature_i in range(generic.shape[1])
        )
    assert blocks, "No readout feature blocks were built."
    return scipy.sparse.hstack(blocks, format="csr", dtype=np.float32), feature_meta


def _load_baseline(path: Path) -> dict[str, dict[str, Any]]:
    baseline = {}
    for row in _read_jsonl(path):
        key = f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"
        baseline[key] = row
    return baseline


def _baseline_scores(row: dict[str, Any] | None) -> dict[str, float] | None:
    if row is None:
        return None
    if row.get("label_scores"):
        return row["label_scores"]
    if row.get("decode_strategy") != "yes_no":
        return None
    schema = row.get("schema") or {}
    primary_labels = (schema.get("primary_field") or {}).get("labels") or []
    if len(primary_labels) != 2:
        return None
    neg_label, pos_label = primary_labels[0], primary_labels[-1]
    gap = max(float(row.get("conf_gap") or 1e-9), 1e-9)
    if row.get("is_positive"):
        return {pos_label: gap, neg_label: -gap}
    return {pos_label: -gap, neg_label: gap}


def _ensemble_sweep(
    sample_keys: list[str],
    gold: list[str],
    probe_proba: list[dict[str, float]],
    baseline: dict[str, dict[str, Any]],
    weights: list[float],
) -> dict[str, Any]:
    valid = []
    for key, label, proba in zip(sample_keys, gold, probe_proba):
        base_scores = _baseline_scores(baseline.get(key))
        if base_scores is not None:
            valid.append((key, label, proba, base_scores))
    sweep = []
    for weight in weights:
        preds = []
        labels = []
        for _, label, proba, base_scores in valid:
            base_proba = _softmax(base_scores)
            combined = {
                cls: weight * proba.get(cls, 0.0) + (1.0 - weight) * base_proba.get(cls, 0.0)
                for cls in LABELS
            }
            preds.append(max(combined, key=combined.__getitem__))
            labels.append(label)
        metrics = _per_class_metrics(labels, preds)
        sweep.append({"probe_weight": weight, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]})
    best = max(sweep, key=lambda row: row["macro_f1"]) if sweep else None
    return {"n": len(valid), "best": best, "sweep": sweep}


def _pair_metrics(
    pairs: list[tuple[str, str]],
    pred_by_key: dict[str, str],
    proba_by_key: dict[str, dict[str, float]],
) -> dict[str, Any]:
    rows = []
    for hateful_key, benign_key in pairs:
        if hateful_key not in pred_by_key or benign_key not in pred_by_key:
            continue
        hateful_pred = pred_by_key[hateful_key]
        benign_pred = pred_by_key[benign_key]
        hateful_prob = proba_by_key[hateful_key].get("hateful", 0.0)
        benign_prob = proba_by_key[benign_key].get("hateful", 0.0)
        rows.append({
            "hateful_key": hateful_key,
            "benign_key": benign_key,
            "both_correct": hateful_pred == "hateful" and benign_pred == "not hateful",
            "separated": hateful_pred != benign_pred,
            "positive_margin": hateful_prob > benign_prob,
            "prob_gap": hateful_prob - benign_prob,
        })
    if not rows:
        return {"n_pairs": 0, "rows": []}
    return {
        "n_pairs": len(rows),
        "both_correct_rate": round(float(np.mean([row["both_correct"] for row in rows])), 6),
        "separated_rate": round(float(np.mean([row["separated"] for row in rows])), 6),
        "positive_margin_rate": round(float(np.mean([row["positive_margin"] for row in rows])), 6),
        "median_prob_gap": round(float(np.median([row["prob_gap"] for row in rows])), 6),
        "rows": rows,
    }


def _plot_sweeps(metrics: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    threshold_sweep = metrics["threshold_sweep"]
    axes[0].plot([row["threshold"] for row in threshold_sweep], [row["macro_f1"] for row in threshold_sweep], marker="o")
    axes[0].set_xlabel("hateful threshold")
    axes[0].set_ylabel("macro F1")
    axes[0].set_title("Readout threshold calibration")
    axes[0].grid(alpha=0.2)

    ensemble_sweep = metrics["ensemble"]["sweep"]
    axes[1].plot([row["probe_weight"] for row in ensemble_sweep], [row["macro_f1"] for row in ensemble_sweep], marker="o", color="#4c78a8")
    axes[1].set_xlabel("readout weight")
    axes[1].set_ylabel("macro F1")
    axes[1].set_title("Readout + baseline ensemble")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "confounder_readout_sweeps.png", dpi=150)
    plt.close(fig)


def run_readout(cfg: FHMConfounderReadoutConfig) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
    import scipy.sparse

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cache = Path(cfg.train_cache_dir)
    val_cache = Path(cfg.val_cache_dir)
    metadata_path = Path(cfg.fhm_confounder_metadata_path)
    baseline_path = Path(cfg.baseline_predictions_path)
    assert train_cache.exists(), f"Train cache missing: {train_cache}"
    assert val_cache.exists(), f"Validation cache missing: {val_cache}"
    assert metadata_path.exists(), f"Metadata missing: {metadata_path}"
    assert baseline_path.exists(), f"Baseline predictions missing: {baseline_path}"

    device = resolve_visible_device(cfg.device)
    logger.info("Loading SAE from %s", cfg.residual_sae_path)
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)
    metadata = _load_fhm_confounder_metadata(metadata_path)

    train_rows = _load_fhm_rows(train_cache, {"train"})
    val_rows = _load_fhm_rows(val_cache, {"validation", "val", "dev"}, cache_tag=True)
    logger.info("FHM rows: train=%d val=%d", len(train_rows), len(val_rows))

    extract_common = dict(
        aggregation=cfg.aggregation,
        batch_size=cfg.batch_size,
        device=device,
        n_features=sae.num_latents,
    )
    train_feature_sets = {}
    val_feature_sets = {}
    for token_selection in cfg.token_selections:
        logger.info("Extracting %s train/val features", token_selection)
        train_feature_sets[token_selection] = _extract_features(
            train_rows, train_cache, sae, stats, token_selection=token_selection, **extract_common,
        )
        val_feature_sets[token_selection] = _extract_features_multi(
            val_rows, sae, stats, token_selection=token_selection, **extract_common,
        )

    train_keys = _common_keys(train_feature_sets, cfg.token_selections[0])
    val_keys = _common_keys(val_feature_sets, cfg.token_selections[0])
    assert train_keys and val_keys, "No common FHM keys across requested token selections."
    logger.info("Common keys: train=%d val=%d", len(train_keys), len(val_keys))

    audits: dict[str, dict[str, Any]] = {token_selection: {} for token_selection in cfg.token_selections}
    train_pairs_by_source = {}
    val_pairs_by_source = {}
    train_key_set = set(train_keys)
    val_key_set = set(val_keys)
    for source in cfg.pair_sources:
        train_pairs_by_source[source] = _build_pairs(train_rows, metadata, train_key_set, source)
        val_pairs_by_source[source] = _build_pairs(val_rows, metadata, val_key_set, source)
    for token_selection in cfg.token_selections:
        keys, csr = train_feature_sets[token_selection]
        key_to_idx = {key: idx for idx, key in enumerate(keys)}
        for source in cfg.pair_sources:
            audits[token_selection][source] = _feature_stats(
                csr, key_to_idx, train_pairs_by_source[source], cfg.top_k_features_per_block,
            )

    X_train_all, feature_meta = _build_readout_matrix(train_feature_sets, train_keys, audits, cfg)
    X_val, _ = _build_readout_matrix(val_feature_sets, val_keys, audits, cfg)
    train_row_by_key = {_row_sample_key(row): row for row in train_rows}
    val_row_by_key = {_row_sample_key(row): row for row in val_rows}
    y_train = [_row_gold_label(train_row_by_key[key]) for key in train_keys]
    y_val = [_row_gold_label(val_row_by_key[key]) for key in val_keys]
    X_train = X_train_all
    pair_delta_stats = {source: len(pairs) for source, pairs in train_pairs_by_source.items()}

    if cfg.include_pair_delta_rows:
        key_to_compact_idx = {key: idx for idx, key in enumerate(train_keys)}
        delta_rows = []
        delta_labels = []
        seen_pairs: set[tuple[str, str]] = set()
        for source in cfg.pair_sources:
            for hateful_key, benign_key in train_pairs_by_source[source]:
                pair_key = tuple(sorted((hateful_key, benign_key)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                h_vec = X_train_all[key_to_compact_idx[hateful_key]]
                b_vec = X_train_all[key_to_compact_idx[benign_key]]
                delta_rows.append(h_vec - b_vec)
                delta_labels.append("hateful")
                delta_rows.append(b_vec - h_vec)
                delta_labels.append("not hateful")
        if delta_rows:
            X_train = scipy.sparse.vstack([X_train_all, *delta_rows], format="csr")
            y_train = list(y_train) + delta_labels
        logger.info("Pair delta rows: pairs=%d synthetic_rows=%d", len(seen_pairs), len(delta_labels))

    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    classes = le.classes_.tolist()
    assert set(classes) == set(LABELS), f"Unexpected classes: {classes}"

    if cfg.classifier_type == "mlp":
        from sklearn.neural_network import MLPClassifier
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
        clf.fit(X_train_scaled.toarray(), y_train_enc)
        y_pred_enc = clf.predict(X_val_scaled.toarray())
        y_proba = clf.predict_proba(X_val_scaled.toarray())
    else:
        clf = LogisticRegression(
            C=cfg.C,
            max_iter=cfg.max_iter,
            solver="liblinear",
            class_weight=cfg.class_weight if cfg.class_weight != "None" else None,
        )
        clf.fit(X_train_scaled, y_train_enc)
        y_pred_enc = clf.predict(X_val_scaled)
        y_proba = clf.predict_proba(X_val_scaled)

    y_pred = le.inverse_transform(y_pred_enc).tolist()
    raw_metrics = _per_class_metrics(y_val, y_pred)
    threshold_sweep = []
    for t_int in range(10, 91):
        threshold = t_int / 100.0
        pos_idx = classes.index("hateful")
        preds = ["hateful" if prob[pos_idx] >= threshold else "not hateful" for prob in y_proba]
        metrics = _per_class_metrics(y_val, preds)
        threshold_sweep.append({"threshold": threshold, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]})
    best_t, best_f1, best_cal_preds = _calibrate_threshold(y_val, y_proba, classes, "hateful")
    calibrated_metrics = _per_class_metrics(y_val, best_cal_preds)
    calibrated_metrics["threshold"] = round(best_t, 2)

    proba_rows = [
        {classes[col_i]: round(float(y_proba[row_i, col_i]), 6) for col_i in range(len(classes))}
        for row_i in range(len(val_keys))
    ]
    baseline = _load_baseline(baseline_path)
    baseline_preds = [baseline.get(key, {}).get("prediction_label") for key in val_keys]
    baseline_metrics = _per_class_metrics(
        [gold for gold, pred in zip(y_val, baseline_preds) if pred in set(LABELS)],
        [pred for pred in baseline_preds if pred in set(LABELS)],
    )
    ensemble = _ensemble_sweep(val_keys, y_val, proba_rows, baseline, cfg.ensemble_weights)

    pred_by_key = dict(zip(val_keys, y_pred))
    proba_by_key = dict(zip(val_keys, proba_rows))
    pair_metrics = {
        source: _pair_metrics(val_pairs_by_source[source], pred_by_key, proba_by_key)
        for source in cfg.pair_sources
    }

    metrics = {
        "config": vars(cfg),
        "n_train_real": len(train_keys),
        "n_train_after_delta": int(X_train.shape[0]),
        "n_val": len(val_keys),
        "n_features_compact": int(X_train_all.shape[1]),
        "class_counts_train": dict(Counter(y_train)),
        "class_counts_val": dict(Counter(y_val)),
        "pair_delta_stats": pair_delta_stats,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "baseline_metrics": baseline_metrics,
        "ensemble": ensemble,
        "threshold_sweep": threshold_sweep,
        "pair_metrics": {
            source: {key: value for key, value in source_metrics.items() if key != "rows"}
            for source, source_metrics in pair_metrics.items()
        },
    }
    (output_dir / "confounder_readout_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "selected_confounder_features.json").write_text(json.dumps(feature_meta, indent=2))
    with (output_dir / "probe_val_predictions.jsonl").open("w") as fd:
        for key, gold, pred, proba in zip(val_keys, y_val, y_pred, proba_rows):
            base_row = baseline.get(key)
            fd.write(json.dumps({
                "sample_key": key,
                "task_key": TASK_KEY,
                "gold_label": gold,
                "probe_pred": pred,
                "probe_proba": proba,
                "baseline_pred": base_row.get("prediction_label") if base_row else None,
                "baseline_label_scores": base_row.get("label_scores") if base_row else None,
                "probe_correct": pred == gold,
                "baseline_correct": (base_row.get("prediction_label") == gold) if base_row else None,
            }) + "\n")
    with (output_dir / "validation_pair_metrics.jsonl").open("w") as fd:
        for source, source_metrics in pair_metrics.items():
            for row in source_metrics.get("rows", []):
                fd.write(json.dumps({"source": source, **row}) + "\n")

    if cfg.plot:
        _plot_sweeps(metrics, output_dir)

    logger.info(
        "FHM confounder readout: raw_f1=%.4f calibrated_f1=%.4f baseline_f1=%.4f ensemble_f1=%.4f",
        raw_metrics["macro_f1"], calibrated_metrics["macro_f1"],
        baseline_metrics["macro_f1"], ensemble["best"]["macro_f1"] if ensemble["best"] else float("nan"),
    )
    logger.info("Wrote outputs to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FHMConfounderReadoutConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run_readout(cfg)


if __name__ == "__main__":
    main()