

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.analyze_fhm_confounder_features import _build_pairs
from gemma3_experiments.launch.residual.probe_fhm_confounder_readout import LABELS, TASK_KEY, _pair_metrics
from gemma3_experiments.launch.residual.probe_validate import (
    _build_fhm_metadata_hard_negative_pairs,
    _calibrate_threshold,
    _extract_features,
    _load_fhm_confounder_metadata,
    _per_class_metrics,
    _read_jsonl,
    _row_gold_label,
    _row_sample_key,
)
from gemma3_experiments.residual_sae import load_residual_sae, load_residual_stats, resolve_visible_device

logger = logging.getLogger(__name__)


@dataclass
class LayerSpec:
    name: str = ""
    residual_sae_path: str = ""
    train_cache_dir: str = ""
    val_cache_dir: str = ""


@dataclass
class FHMLayerConcatConfig:
    layer9_name: str = "layer9"
    layer9_residual_sae_path: str = ""
    layer9_train_cache_dir: str = ""
    layer9_val_cache_dir: str = ""

    layer17_name: str = "layer17"
    layer17_residual_sae_path: str = ""
    layer17_train_cache_dir: str = ""
    layer17_val_cache_dir: str = ""

    layer22_name: str = "layer22"
    layer22_residual_sae_path: str = ""
    layer22_train_cache_dir: str = ""
    layer22_val_cache_dir: str = ""

    layer29_name: str = "layer29"
    layer29_residual_sae_path: str = ""
    layer29_train_cache_dir: str = ""
    layer29_val_cache_dir: str = ""

    baseline_predictions_path: str = ""
    fhm_confounder_metadata_path: str = ""
    output_dir: str = ""

    token_selection: str = "all"
    aggregation: str = "max"
    batch_size: int = 256
    include_pair_delta_rows: bool = True
    pair_sources: list[str] = field(default_factory=lambda: ["pseudo_img", "pseudo_text"])

    classifier_type: str = "mlp"  # mlp | logistic
    C: float = 1.0
    max_iter: int = 5000
    class_weight: str = "balanced"
    mlp_hidden_size: int = 512
    mlp_alpha: float = 1e-4
    mlp_max_iter: int = 200

    device: str = "cuda:0"


def _load_fhm_rows(cache_dir: Path, split_names: set[str]) -> list[dict[str, Any]]:
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
        rows.append(row)
    return rows


def _align_csr(keys: list[str], csr: Any, common_keys: list[str]) -> Any:
    key_to_idx = {key: idx for idx, key in enumerate(keys)}
    return csr[[key_to_idx[key] for key in common_keys]]


def _common_keys(key_lists: list[list[str]]) -> list[str]:
    common = set(key_lists[0])
    for keys in key_lists[1:]:
        common &= set(keys)
    return [key for key in key_lists[0] if key in common]


def _extract_layer_features(spec: LayerSpec, cfg: FHMLayerConcatConfig, device: str) -> dict[str, Any]:
    train_cache = Path(spec.train_cache_dir)
    val_cache = Path(spec.val_cache_dir)
    assert train_cache.exists(), f"Train cache missing for {spec.name}: {train_cache}"
    assert val_cache.exists(), f"Validation cache missing for {spec.name}: {val_cache}"

    logger.info("Loading %s SAE from %s", spec.name, spec.residual_sae_path)
    sae = load_residual_sae(spec.residual_sae_path, device=device)
    stats = load_residual_stats(spec.residual_sae_path)
    train_rows = _load_fhm_rows(train_cache, {"train"})
    val_rows = _load_fhm_rows(val_cache, {"validation", "val", "dev"})
    logger.info("%s rows: train=%d val=%d", spec.name, len(train_rows), len(val_rows))

    extract_common = dict(
        token_selection=cfg.token_selection,
        aggregation=cfg.aggregation,
        batch_size=cfg.batch_size,
        device=device,
        n_features=sae.num_latents,
    )
    train_keys, train_csr = _extract_features(train_rows, train_cache, sae, stats, **extract_common)
    val_keys, val_csr = _extract_features(val_rows, val_cache, sae, stats, **extract_common)
    return {
        "spec": spec,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "train_keys": train_keys,
        "val_keys": val_keys,
        "train_csr": train_csr,
        "val_csr": val_csr,
        "n_features": int(train_csr.shape[1]),
    }


def _load_baseline(path: Path) -> dict[str, dict[str, Any]]:
    baseline = {}
    for row in _read_jsonl(path):
        key = f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"
        baseline[key] = row
    return baseline


def _baseline_metrics(keys: list[str], y_val: list[str], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gold = []
    pred = []
    for key, label in zip(keys, y_val):
        base_pred = baseline.get(key, {}).get("prediction_label")
        if base_pred in set(LABELS):
            gold.append(label)
            pred.append(base_pred)
    return _per_class_metrics(gold, pred)


def _layer_specs_from_config(cfg: FHMLayerConcatConfig) -> list[LayerSpec]:
    specs = [
        LayerSpec(cfg.layer9_name, cfg.layer9_residual_sae_path, cfg.layer9_train_cache_dir, cfg.layer9_val_cache_dir),
        LayerSpec(cfg.layer17_name, cfg.layer17_residual_sae_path, cfg.layer17_train_cache_dir, cfg.layer17_val_cache_dir),
        LayerSpec(cfg.layer22_name, cfg.layer22_residual_sae_path, cfg.layer22_train_cache_dir, cfg.layer22_val_cache_dir),
        LayerSpec(cfg.layer29_name, cfg.layer29_residual_sae_path, cfg.layer29_train_cache_dir, cfg.layer29_val_cache_dir),
    ]
    return [spec for spec in specs if spec.residual_sae_path]


def run_layer_concat(cfg: FHMLayerConcatConfig) -> None:
    import scipy.sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

    layer_specs = _layer_specs_from_config(cfg)
    assert len(layer_specs) >= 2, "Expected at least two configured residual-SAE layers."
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(cfg.baseline_predictions_path)
    metadata_path = Path(cfg.fhm_confounder_metadata_path)
    assert baseline_path.exists(), f"Baseline predictions missing: {baseline_path}"
    assert metadata_path.exists(), f"FHM metadata missing: {metadata_path}"

    device = resolve_visible_device(cfg.device)
    layers = [_extract_layer_features(spec, cfg, device) for spec in layer_specs]
    train_keys = _common_keys([layer["train_keys"] for layer in layers])
    val_keys = _common_keys([layer["val_keys"] for layer in layers])
    assert train_keys and val_keys, "No common train/validation keys across layer features."
    logger.info("Common keys across layers: train=%d val=%d", len(train_keys), len(val_keys))

    X_train_all = scipy.sparse.hstack(
        [_align_csr(layer["train_keys"], layer["train_csr"], train_keys) for layer in layers],
        format="csr",
        dtype=np.float32,
    )
    X_val = scipy.sparse.hstack(
        [_align_csr(layer["val_keys"], layer["val_csr"], val_keys) for layer in layers],
        format="csr",
        dtype=np.float32,
    )
    train_row_by_key = {_row_sample_key(row): row for row in layers[0]["train_rows"]}
    val_row_by_key = {_row_sample_key(row): row for row in layers[0]["val_rows"]}
    y_train = [_row_gold_label(train_row_by_key[key]) for key in train_keys]
    y_val = [_row_gold_label(val_row_by_key[key]) for key in val_keys]

    X_train = X_train_all
    hard_negative_stats = None
    if cfg.include_pair_delta_rows:
        train_key_to_idx = {key: idx for idx, key in enumerate(train_keys)}
        delta_csr, delta_labels, hard_negative_stats = _build_fhm_metadata_hard_negative_pairs(
            layers[0]["train_rows"],
            X_train_all,
            train_key_to_idx,
            metadata_path,
            cfg.pair_sources,
        )
        if delta_csr.shape[0]:
            X_train = scipy.sparse.vstack([X_train_all, delta_csr], format="csr")
            y_train = list(y_train) + delta_labels
            logger.info("Pair delta rows: train %d -> %d", len(train_keys), X_train.shape[0])

    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    classes = le.classes_.tolist()
    assert set(classes) == set(LABELS), f"Unexpected classes: {classes}"

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
    pos_idx = classes.index("hateful")
    for t_int in range(10, 91):
        threshold = t_int / 100.0
        preds = ["hateful" if prob[pos_idx] >= threshold else "not hateful" for prob in y_proba]
        metrics = _per_class_metrics(y_val, preds)
        threshold_sweep.append({"threshold": threshold, "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]})
    best_t, _, best_cal_preds = _calibrate_threshold(y_val, y_proba, classes, "hateful")
    calibrated_metrics = _per_class_metrics(y_val, best_cal_preds)
    calibrated_metrics["threshold"] = round(best_t, 2)

    proba_rows = [
        {classes[col_i]: round(float(y_proba[row_i, col_i]), 6) for col_i in range(len(classes))}
        for row_i in range(len(val_keys))
    ]
    baseline = _load_baseline(baseline_path)
    baseline_metrics = _baseline_metrics(val_keys, y_val, baseline)

    metadata = _load_fhm_confounder_metadata(metadata_path)
    val_key_set = set(val_keys)
    val_pairs_by_source = {
        source: _build_pairs(layers[0]["val_rows"], metadata, val_key_set, source)
        for source in cfg.pair_sources
    }
    pred_by_key = dict(zip(val_keys, y_pred))
    proba_by_key = dict(zip(val_keys, proba_rows))
    pair_metrics = {
        source: _pair_metrics(val_pairs_by_source[source], pred_by_key, proba_by_key)
        for source in cfg.pair_sources
    }

    metrics = {
        "config": {
            **vars(cfg),
            "layer_specs": [vars(spec) for spec in layer_specs],
        },
        "task_key": TASK_KEY,
        "n_train_real": len(train_keys),
        "n_train_after_delta": int(X_train.shape[0]),
        "n_val": len(val_keys),
        "n_features_total": int(X_train_all.shape[1]),
        "layer_feature_dims": {layer["spec"].name: layer["n_features"] for layer in layers},
        "class_counts_train": dict(Counter(y_train)),
        "class_counts_val": dict(Counter(y_val)),
        "hard_negative_stats": hard_negative_stats,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "baseline_metrics": baseline_metrics,
        "threshold_sweep": threshold_sweep,
        "pair_metrics": {
            source: {key: value for key, value in source_metrics.items() if key != "rows"}
            for source, source_metrics in pair_metrics.items()
        },
    }
    (output_dir / "layer_concat_metrics.json").write_text(json.dumps(metrics, indent=2))
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

    logger.info(
        "FHM layer concat readout: raw_f1=%.4f calibrated_f1=%.4f baseline_f1=%.4f",
        raw_metrics["macro_f1"], calibrated_metrics["macro_f1"], baseline_metrics["macro_f1"],
    )
    logger.info("Wrote outputs to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FHMLayerConcatConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run_layer_concat(cfg)


if __name__ == "__main__":
    main()
