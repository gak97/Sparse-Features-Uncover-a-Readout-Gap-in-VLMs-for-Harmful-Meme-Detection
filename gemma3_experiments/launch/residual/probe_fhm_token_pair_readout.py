

import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.analyze_fhm_confounder_features import (
    _build_pairs,
    _feature_stats,
)
from gemma3_experiments.launch.residual.probe_fhm_confounder_readout import (
    LABELS,
    TASK_KEY,
    _ensemble_sweep,
    _load_baseline,
    _load_fhm_rows,
    _pair_metrics,
    _plot_sweeps,
)
from gemma3_experiments.launch.residual.probe_validate import (
    _calibrate_threshold,
    _dense_acts,
    _extract_features,
    _extract_features_multi,
    _load_fhm_confounder_metadata,
    _parse_interaction_pair,
    _per_class_metrics,
    _role_mask,
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
class FHMTokenPairReadoutConfig:
    residual_sae_path: str = ""
    train_cache_dir: str = ""
    val_cache_dir: str = ""
    baseline_predictions_path: str = ""
    output_dir: str = ""
    fhm_confounder_metadata_path: str = ""

    audit_token_selections: list[str] = field(default_factory=lambda: ["all", "image", "prompt", "generated"])
    token_roles: list[str] = field(default_factory=lambda: ["image", "prompt", "generated", "all"])
    pair_sources: list[str] = field(default_factory=lambda: ["pseudo_img", "pseudo_text"])
    role_interaction_pairs: list[str] = field(default_factory=lambda: ["image:prompt", "image:generated", "prompt:generated"])
    token_stats: list[str] = field(default_factory=lambda: ["max", "mean", "active_frac", "top2_mean"])
    interaction_ops: list[str] = field(default_factory=lambda: ["diff", "absdiff", "product"])

    aggregation: str = "max"
    batch_size: int = 256
    top_k_features_per_block: int = 75
    selected_features_per_block: int = 50
    projection_top_k: list[int] = field(default_factory=lambda: [5, 10, 20, 50])
    include_projection_columns: bool = True
    include_role_token_counts: bool = True
    include_generic_all_features: bool = True
    include_pair_delta_rows: bool = True

    # Quadratic within-block: append I², P², G² (element-wise squared max-pool
    # activations for each role) as additional features.  Allows the linear
    # classifier to implement a diagonal quadratic boundary without any cross-
    # modal interaction.  Use as a control against the bilinear cross-modal term.
    quadratic_blocks: bool = False

    # Splits to load from val_cache_dir.  Default covers all standard validation
    # names; override to ["test"] for test-split transfer evaluation.
    eval_splits: list[str] = field(default_factory=lambda: ["validation", "val", "dev"])

    classifier_type: str = "logistic"  # logistic | mlp
    C: float = 1.0
    max_iter: int = 5000
    class_weight: str = "balanced"
    mlp_hidden_size: int = 256
    mlp_alpha: float = 1e-3
    mlp_max_iter: int = 300

    ensemble_weights: list[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ])
    plot: bool = True
    device: str = "cuda:0"


def _required_audit_selections(cfg: FHMTokenPairReadoutConfig) -> list[str]:
    selections = list(cfg.audit_token_selections)
    if cfg.include_generic_all_features:
        selections.append("all")
    return list(dict.fromkeys(selections))


def _align_csr(keys: list[str], csr: Any, common_keys: list[str]) -> Any:
    key_to_idx = {key: idx for idx, key in enumerate(keys)}
    return csr[[key_to_idx[key] for key in common_keys]]


def _selected_feature_indices(audits: dict[str, dict[str, Any]], cfg: FHMTokenPairReadoutConfig) -> list[int]:
    feature_indices: set[int] = set()
    for token_selection in cfg.audit_token_selections:
        for source in cfg.pair_sources:
            for row in audits[token_selection][source]["top_features"][:cfg.selected_features_per_block]:
                feature_indices.add(int(row["feature_idx"]))
    selected = sorted(feature_indices)
    assert selected, "No token-level features selected from audit results."
    return selected


def _projection_specs(
    audits: dict[str, dict[str, Any]],
    selected_pos: dict[int, int],
    cfg: FHMTokenPairReadoutConfig,
) -> list[dict[str, Any]]:
    specs = []
    if not cfg.include_projection_columns:
        return specs
    for token_selection in cfg.audit_token_selections:
        for source in cfg.pair_sources:
            top_features = audits[token_selection][source]["top_features"]
            for top_k in cfg.projection_top_k:
                rows = top_features[:top_k]
                positions = [selected_pos[int(row["feature_idx"])] for row in rows if int(row["feature_idx"]) in selected_pos]
                signs = [1.0 if row["direction"] == "hateful_up" else -1.0 for row in rows if int(row["feature_idx"]) in selected_pos]
                if not positions:
                    continue
                specs.append({
                    "token_selection": token_selection,
                    "source": source,
                    "top_k": int(len(positions)),
                    "positions": np.array(positions, dtype=np.int32),
                    "signs": np.array(signs, dtype=np.float32),
                })
    return specs


def _feature_meta(
    selected_features: list[int],
    projection_specs: list[dict[str, Any]],
    cfg: FHMTokenPairReadoutConfig,
) -> list[dict[str, Any]]:
    meta = []
    for role in cfg.token_roles:
        for stat in cfg.token_stats:
            for feature_idx in selected_features:
                meta.append({"kind": "role_stat", "role": role, "stat": stat, "feature_idx": feature_idx})
        if cfg.include_role_token_counts:
            meta.append({"kind": "role_token_count_log", "role": role})

    for pair in cfg.role_interaction_pairs:
        left, right = _parse_interaction_pair(pair)
        for op in cfg.interaction_ops:
            for feature_idx in selected_features:
                meta.append({"kind": "role_interaction", "left": left, "right": right, "op": op, "feature_idx": feature_idx})

    for spec_i, spec in enumerate(projection_specs):
        for role in cfg.token_roles:
            for stat in ("max", "mean"):
                meta.append({
                    "kind": "role_projection",
                    "projection_idx": spec_i,
                    "role": role,
                    "stat": stat,
                    "token_selection": spec["token_selection"],
                    "source": spec["source"],
                    "top_k": spec["top_k"],
                })
        for pair in cfg.role_interaction_pairs:
            left, right = _parse_interaction_pair(pair)
            for op in cfg.interaction_ops:
                meta.append({
                    "kind": "projection_interaction",
                    "projection_idx": spec_i,
                    "left": left,
                    "right": right,
                    "op": op,
                    "token_selection": spec["token_selection"],
                    "source": spec["source"],
                    "top_k": spec["top_k"],
                })
    if cfg.include_generic_all_features:
        meta.append({"kind": "generic_all_sae_block", "n_features": "sae.num_latents"})
    return meta


def _role_acts(
    sae,
    residual: torch.Tensor,
    roles: torch.Tensor,
    stats,
    *,
    token_selection: str,
    selected_feature_i: torch.Tensor,
    batch_size: int,
    device: str,
) -> torch.Tensor | None:
    mask = _role_mask(roles, token_selection)
    if not bool(mask.any()):
        return None
    selected = residual[mask].to(device=device, dtype=torch.float32)
    if stats is not None:
        selected = stats.normalize(selected)
    chunks = []
    with torch.inference_mode():
        for start in range(0, selected.shape[0], batch_size):
            acts = _dense_acts(sae, selected[start: start + batch_size])
            chunks.append(acts[:, selected_feature_i].float().cpu())
    return torch.cat(chunks, dim=0) if chunks else None


def _summarise_acts(acts: torch.Tensor | None, n_features: int) -> tuple[dict[str, np.ndarray], float]:
    zeros = np.zeros(n_features, dtype=np.float32)
    if acts is None or acts.numel() == 0:
        return {"max": zeros, "mean": zeros, "active_frac": zeros, "top2_mean": zeros}, 0.0

    max_vals = acts.max(dim=0).values.numpy().astype(np.float32)
    mean_vals = acts.mean(dim=0).numpy().astype(np.float32)
    active_frac = (acts > 0).float().mean(dim=0).numpy().astype(np.float32)
    top_k = min(2, acts.shape[0])
    top2_mean = acts.topk(top_k, dim=0).values.mean(dim=0).numpy().astype(np.float32)
    return {
        "max": max_vals,
        "mean": mean_vals,
        "active_frac": active_frac,
        "top2_mean": top2_mean,
    }, float(math.log1p(acts.shape[0]))


def _projection_value(summary: dict[str, np.ndarray], spec: dict[str, Any], stat: str) -> float:
    values = summary[stat][spec["positions"]]
    return float(values @ spec["signs"] / math.sqrt(len(spec["positions"])))


def _interaction(left: np.ndarray, right: np.ndarray, op: str) -> np.ndarray:
    if op == "diff":
        return left - right
    if op == "absdiff":
        return np.abs(left - right)
    if op == "product":
        return left * right
    raise ValueError(f"Unsupported interaction op: {op!r}")


def _sample_vector(
    role_summaries: dict[str, dict[str, np.ndarray]],
    role_counts: dict[str, float],
    projection_specs: list[dict[str, Any]],
    cfg: FHMTokenPairReadoutConfig,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for role in cfg.token_roles:
        for stat in cfg.token_stats:
            parts.append(role_summaries[role][stat])
        if cfg.include_role_token_counts:
            parts.append(np.array([role_counts[role]], dtype=np.float32))

    for pair in cfg.role_interaction_pairs:
        left, right = _parse_interaction_pair(pair)
        for op in cfg.interaction_ops:
            parts.append(_interaction(role_summaries[left]["max"], role_summaries[right]["max"], op))

    # Quadratic within-block: I², P², G² (control for bilinear cross-modal)
    if cfg.quadratic_blocks:
        for role in ("image", "prompt", "generated"):
            if role in role_summaries:
                parts.append(role_summaries[role]["max"] ** 2)

    for spec in projection_specs:
        role_projection: dict[tuple[str, str], float] = {}
        for role in cfg.token_roles:
            for stat in ("max", "mean"):
                value = _projection_value(role_summaries[role], spec, stat)
                parts.append(np.array([value], dtype=np.float32))
                role_projection[(role, stat)] = value
        for pair in cfg.role_interaction_pairs:
            left, right = _parse_interaction_pair(pair)
            for op in cfg.interaction_ops:
                left_value = np.array([role_projection[(left, "max")]], dtype=np.float32)
                right_value = np.array([role_projection[(right, "max")]], dtype=np.float32)
                parts.append(_interaction(left_value, right_value, op))
    return np.concatenate(parts).astype(np.float32)


def _extract_token_summary_features(
    rows: list[dict[str, Any]],
    cache_dir: Path,
    sae,
    stats,
    *,
    selected_features: list[int],
    projection_specs: list[dict[str, Any]],
    cfg: FHMTokenPairReadoutConfig,
    device: str,
) -> tuple[list[str], Any]:
    import scipy.sparse

    by_shard: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row_cache_dir = row.get("_cache_dir", str(cache_dir))
        by_shard[(row_cache_dir, row["shard_file"])].append(row)

    selected_feature_i = torch.tensor(selected_features, dtype=torch.long, device=device)
    n_features = len(selected_features)
    sample_keys: list[str] = []
    vectors: list[np.ndarray] = []

    for shard_idx, ((cache_dir_str, shard_file), shard_rows) in enumerate(sorted(by_shard.items())):
        shard_path = Path(cache_dir_str) / shard_file
        assert shard_path.exists(), f"Shard missing: {shard_path}"
        shard_data = load_file(str(shard_path))
        if (shard_idx + 1) % 100 == 0 or shard_idx == 0:
            logger.info(
                "Extracting token summaries shard %d/%d (%d rows, %d extracted so far)",
                shard_idx + 1, len(by_shard), len(shard_rows), len(vectors),
            )

        for row in sorted(shard_rows, key=lambda r: int(r["offset"])):
            offset, length = int(row["offset"]), int(row["length"])
            residual = shard_data["residual"][offset: offset + length]
            roles = shard_data["token_roles"][offset: offset + length]
            role_summaries = {}
            role_counts = {}
            for role in cfg.token_roles:
                acts = _role_acts(
                    sae, residual, roles, stats,
                    token_selection=role,
                    selected_feature_i=selected_feature_i,
                    batch_size=cfg.batch_size,
                    device=device,
                )
                role_summaries[role], role_counts[role] = _summarise_acts(acts, n_features)
            vectors.append(_sample_vector(role_summaries, role_counts, projection_specs, cfg))
            sample_keys.append(_row_sample_key(row))

    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 0), dtype=np.float32)
    logger.info("Token summary extraction complete: %d rows, %d features", matrix.shape[0], matrix.shape[1])
    return sample_keys, scipy.sparse.csr_matrix(matrix, dtype=np.float32)


def _append_pair_delta_rows(X_train_all: Any, y_train: list[str], train_keys: list[str], pairs_by_source: dict[str, list[tuple[str, str]]]) -> tuple[Any, list[str], int, int]:
    import scipy.sparse

    key_to_idx = {key: idx for idx, key in enumerate(train_keys)}
    delta_rows = []
    delta_labels = []
    seen_pairs: set[tuple[str, str]] = set()
    for pairs in pairs_by_source.values():
        for hateful_key, benign_key in pairs:
            if hateful_key not in key_to_idx or benign_key not in key_to_idx:
                continue
            pair_key = tuple(sorted((hateful_key, benign_key)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            h_vec = X_train_all[key_to_idx[hateful_key]]
            b_vec = X_train_all[key_to_idx[benign_key]]
            delta_rows.append(h_vec - b_vec)
            delta_labels.append("hateful")
            delta_rows.append(b_vec - h_vec)
            delta_labels.append("not hateful")
    if not delta_rows:
        return X_train_all, y_train, 0, 0
    X_train = scipy.sparse.vstack([X_train_all, *delta_rows], format="csr")
    return X_train, list(y_train) + delta_labels, len(seen_pairs), len(delta_labels)


def run_token_pair_readout(cfg: FHMTokenPairReadoutConfig) -> None:
    import scipy.sparse
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

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
    val_rows = _load_fhm_rows(val_cache, set(cfg.eval_splits), cache_tag=True)
    logger.info("FHM rows: train=%d val=%d", len(train_rows), len(val_rows))

    extract_common = dict(
        aggregation=cfg.aggregation,
        batch_size=cfg.batch_size,
        device=device,
        n_features=sae.num_latents,
    )
    train_feature_sets = {}
    val_feature_sets = {}
    for token_selection in _required_audit_selections(cfg):
        logger.info("Extracting pooled %s train/val features", token_selection)
        train_feature_sets[token_selection] = _extract_features(
            train_rows, train_cache, sae, stats, token_selection=token_selection, **extract_common,
        )
        val_feature_sets[token_selection] = _extract_features_multi(
            val_rows, sae, stats, token_selection=token_selection, **extract_common,
        )

    pooled_train_keys = train_feature_sets[cfg.audit_token_selections[0]][0]
    available_train_keys = set(pooled_train_keys)
    train_pairs_by_source = {
        source: _build_pairs(train_rows, metadata, available_train_keys, source)
        for source in cfg.pair_sources
    }

    audits: dict[str, dict[str, Any]] = {token_selection: {} for token_selection in cfg.audit_token_selections}
    for token_selection in cfg.audit_token_selections:
        keys, csr = train_feature_sets[token_selection]
        key_to_idx = {key: idx for idx, key in enumerate(keys)}
        for source in cfg.pair_sources:
            audits[token_selection][source] = _feature_stats(
                csr, key_to_idx, train_pairs_by_source[source], cfg.top_k_features_per_block,
            )

    selected_features = _selected_feature_indices(audits, cfg)
    selected_pos = {feature_idx: idx for idx, feature_idx in enumerate(selected_features)}
    projection_specs = _projection_specs(audits, selected_pos, cfg)
    token_feature_meta = _feature_meta(selected_features, projection_specs, cfg)
    logger.info("Selected %d unique token features and %d projection specs", len(selected_features), len(projection_specs))

    train_token_keys, X_train_token = _extract_token_summary_features(
        train_rows, train_cache, sae, stats,
        selected_features=selected_features,
        projection_specs=projection_specs,
        cfg=cfg,
        device=device,
    )
    val_token_keys, X_val_token = _extract_token_summary_features(
        val_rows, val_cache, sae, stats,
        selected_features=selected_features,
        projection_specs=projection_specs,
        cfg=cfg,
        device=device,
    )

    train_keys = list(train_token_keys)
    val_keys = list(val_token_keys)
    X_train_all = X_train_token
    X_val = X_val_token
    if cfg.include_generic_all_features:
        train_all_keys, train_all_csr = train_feature_sets["all"]
        val_all_keys, val_all_csr = val_feature_sets["all"]
        train_common = [key for key in train_keys if key in set(train_all_keys)]
        val_common = [key for key in val_keys if key in set(val_all_keys)]
        train_token_idx = {key: idx for idx, key in enumerate(train_keys)}
        val_token_idx = {key: idx for idx, key in enumerate(val_keys)}
        train_keys = train_common
        val_keys = val_common
        X_train_all = scipy.sparse.hstack([
            X_train_token[[train_token_idx[key] for key in train_keys]],
            _align_csr(train_all_keys, train_all_csr, train_keys),
        ], format="csr")
        X_val = scipy.sparse.hstack([
            X_val_token[[val_token_idx[key] for key in val_keys]],
            _align_csr(val_all_keys, val_all_csr, val_keys),
        ], format="csr")

    logger.info("Readout matrices: train=%s val=%s", X_train_all.shape, X_val.shape)
    train_row_by_key = {_row_sample_key(row): row for row in train_rows}
    val_row_by_key = {_row_sample_key(row): row for row in val_rows}
    y_train = [_row_gold_label(train_row_by_key[key]) for key in train_keys]
    y_val = [_row_gold_label(val_row_by_key[key]) for key in val_keys]

    train_key_set = set(train_keys)
    val_key_set = set(val_keys)
    train_pairs_by_source = {
        source: _build_pairs(train_rows, metadata, train_key_set, source)
        for source in cfg.pair_sources
    }
    val_pairs_by_source = {
        source: _build_pairs(val_rows, metadata, val_key_set, source)
        for source in cfg.pair_sources
    }

    X_train = X_train_all
    pair_delta_n_pairs = 0
    pair_delta_n_rows = 0
    if cfg.include_pair_delta_rows:
        X_train, y_train, pair_delta_n_pairs, pair_delta_n_rows = _append_pair_delta_rows(
            X_train_all, y_train, train_keys, train_pairs_by_source,
        )
        logger.info("Pair delta rows: pairs=%d synthetic_rows=%d", pair_delta_n_pairs, pair_delta_n_rows)

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
    baseline_gold = []
    baseline_pred = []
    for key, gold in zip(val_keys, y_val):
        pred = baseline.get(key, {}).get("prediction_label")
        if pred in set(LABELS):
            baseline_gold.append(gold)
            baseline_pred.append(pred)
    baseline_metrics = _per_class_metrics(baseline_gold, baseline_pred)
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
        "n_selected_token_features": len(selected_features),
        "n_projection_specs": len(projection_specs),
        "n_features_token_block": int(X_train_token.shape[1]),
        "n_features_total": int(X_train_all.shape[1]),
        "class_counts_train": dict(Counter(y_train)),
        "class_counts_val": dict(Counter(y_val)),
        "pair_delta_stats": {
            "pairs_by_source": {source: len(pairs) for source, pairs in train_pairs_by_source.items()},
            "unique_pairs": pair_delta_n_pairs,
            "synthetic_rows": pair_delta_n_rows,
        },
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
    (output_dir / "token_pair_readout_metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "selected_token_pair_features.json").write_text(json.dumps({
        "selected_features": selected_features,
        "feature_meta": token_feature_meta,
        "projection_specs": [
            {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in spec.items()}
            for spec in projection_specs
        ],
    }, indent=2))
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
        "FHM token pair readout: raw_f1=%.4f calibrated_f1=%.4f baseline_f1=%.4f ensemble_f1=%.4f",
        raw_metrics["macro_f1"], calibrated_metrics["macro_f1"],
        baseline_metrics["macro_f1"], ensemble["best"]["macro_f1"] if ensemble["best"] else float("nan"),
    )
    logger.info("Wrote outputs to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FHMTokenPairReadoutConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run_token_pair_readout(cfg)


if __name__ == "__main__":
    main()