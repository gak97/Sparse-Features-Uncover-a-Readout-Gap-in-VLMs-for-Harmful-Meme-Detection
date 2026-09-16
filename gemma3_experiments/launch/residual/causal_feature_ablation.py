

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.probe_validate import (
    _per_class_metrics,
    _read_jsonl,
)

logger = logging.getLogger(__name__)


@dataclass
class CausalAblationConfig:
    # FHM pair ablation
    fhm_probe_predictions_path: str = ""
    fhm_feature_matrix_path: str = ""
    fhm_confounder_metadata_path: str = ""
    fhm_top_confounder_features: list[int] = field(default_factory=list)
    # Named feature groups for per-group ablation, e.g.
    #   --fhm_feature_groups_entries "harm:2261,14008,1542" "confounder:19691,4163"
    # Each group is ablated independently and pair-margin deltas are reported.
    fhm_feature_groups_entries: list[str] = field(default_factory=list)

    # MMHS hate-type ablation
    mmhs_probe_predictions_path: str = ""
    mmhs_feature_matrix_path: str = ""
    mmhs_top_features_by_class: dict[str, list[int]] = field(default_factory=dict)
    # Alternate CLI format for simple_parsing, e.g.
    # --mmhs_top_features_by_class_entries Racist:1,2,3 Sexist:4,5,6
    mmhs_top_features_by_class_entries: list[str] = field(default_factory=list)

    # Shared
    baseline_predictions_path: str = ""
    output_dir: str = ""
    ablation_method: str = "zero"  # zero | mean | shuffle
    top_k_ablate: int = 20
    device: str = "cpu"


def _load_probe_predictions(path: Path) -> dict[str, dict[str, Any]]:
    by_key = {}
    for row in _read_jsonl(path):
        key = row.get("sample_key") or f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"
        by_key[key] = row
    return by_key


def _load_feature_matrix(path: Path) -> tuple[list[str], Any]:
    import scipy.sparse
    payload = np.load(path, allow_pickle=True)
    if "keys" in payload:
        keys = [str(k) for k in payload["keys"]]
    else:
        # Keys saved as a sibling .npy file (format written by probe_fhm_crosscoder.py).
        keys_path = path.with_name(path.stem + "_keys.npy")
        assert keys_path.exists(), (
            f"Feature matrix has no 'keys' array and no sibling keys file found at {keys_path}"
        )
        keys = [str(k) for k in np.load(str(keys_path), allow_pickle=True)]
    if "X" in payload:
        raw_X = payload["X"]
        if isinstance(raw_X, np.ndarray) and raw_X.dtype == object and raw_X.size == 1:
            raw_X = raw_X.item()
        X = scipy.sparse.csr_matrix(raw_X)
    else:
        # Also support standard scipy.save_npz-like payload components.
        X = scipy.sparse.csr_matrix(
            (payload["data"], payload["indices"], payload["indptr"]),
            shape=tuple(payload["shape"]),
        )
    return keys, X


def _parse_class_feature_entries(entries: list[str]) -> dict[str, list[int]]:
    parsed: dict[str, list[int]] = {}
    for entry in entries:
        token = str(entry).strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                "Invalid mmhs class feature entry: "
                f"{token!r}. Expected format Class:i,j,k"
            )
        cls, raw_features = token.split(":", 1)
        cls = cls.strip()
        if not cls:
            raise ValueError(f"Invalid mmhs class in entry: {token!r}")
        values = [v.strip() for v in raw_features.split(",") if v.strip()]
        parsed[cls] = [int(v) for v in values]
    return parsed


def _normalized_mmhs_top_features(cfg: CausalAblationConfig) -> dict[str, list[int]]:
    if cfg.mmhs_top_features_by_class:
        return {str(k): [int(v) for v in vals] for k, vals in cfg.mmhs_top_features_by_class.items()}
    if cfg.mmhs_top_features_by_class_entries:
        return _parse_class_feature_entries(cfg.mmhs_top_features_by_class_entries)
    return {}


def _ablate_features(X: Any, feature_indices: list[int], method: str, rng: np.random.Generator) -> Any:
    if not feature_indices:
        return X
    n_cols = int(X.shape[1])
    col_idx = np.array(sorted({int(i) for i in feature_indices if 0 <= int(i) < n_cols}), dtype=np.int32)
    if col_idx.size == 0:
        return X

    # Perform structural edits in LIL format; convert back to CSR for downstream predict.
    X = X.copy().tolil()
    if method == "zero":
        X[:, col_idx] = 0
    elif method == "mean":
        means = np.asarray(X[:, col_idx].mean(axis=0)).ravel()
        X[:, col_idx] = X[:, col_idx].toarray() - means[np.newaxis, :]
    elif method == "shuffle":
        for j in col_idx:
            col = X[:, j].toarray().ravel()
            perm = rng.permutation(len(col))
            X[:, j] = col[perm][:, np.newaxis]
    else:
        raise ValueError(f"Unknown ablation method: {method}")
    return X.tocsr()


def _extract_proba_from_preds(
    keys: list[str],
    probe_preds: dict[str, dict[str, Any]],
    classes: list[str],
) -> "np.ndarray | None":
    """Build a (n_keys, n_classes) proba matrix from stored probe_proba dicts.

    Returns None if no stored probabilities are found.
    """
    proba = np.zeros((len(keys), len(classes)), dtype=np.float32)
    found = 0
    for i, key in enumerate(keys):
        row = probe_preds.get(key)
        if row is None:
            continue
        pp = row.get("probe_proba")
        if not pp:
            continue
        found += 1
        for j, cls in enumerate(classes):
            proba[i, j] = float(pp.get(cls, 0.0))
    return proba if found > 0 else None


def _align_proba_to_classes(
    raw_proba: "np.ndarray",
    src_classes: list[str],
    dst_classes: list[str],
) -> "np.ndarray":
    """Re-order raw_proba columns from src_classes order to dst_classes order."""
    aligned = np.zeros((raw_proba.shape[0], len(dst_classes)), dtype=np.float32)
    for j, cls in enumerate(dst_classes):
        if cls in src_classes:
            aligned[:, j] = raw_proba[:, src_classes.index(cls)]
    return aligned


def _compute_pair_margin_metrics(
    proba: "np.ndarray",
    classes: list[str],
    pairs: list[tuple[str, str]],
    key_to_idx: dict[str, int],
) -> dict[str, Any]:
    pass
    pos_label = "hateful"
    if pos_label not in classes:
        return {}
    pos_idx = classes.index(pos_label)
    eps = 1e-7
    margins: list[float] = []
    logit_gaps: list[float] = []
    for h_key, b_key in pairs:
        h_i = key_to_idx.get(h_key)
        b_i = key_to_idx.get(b_key)
        if h_i is None or b_i is None:
            continue
        ph = float(np.clip(proba[h_i, pos_idx], eps, 1.0 - eps))
        pb = float(np.clip(proba[b_i, pos_idx], eps, 1.0 - eps))
        margins.append(ph - pb)
        logit_gaps.append(
            (np.log(ph) - np.log(1.0 - ph)) - (np.log(pb) - np.log(1.0 - pb))
        )
    if not margins:
        return {"n_pairs": 0}
    m = np.array(margins, dtype=np.float64)
    lg = np.array(logit_gaps, dtype=np.float64)
    return {
        "n_pairs": len(m),
        "mean_pair_margin": round(float(np.mean(m)), 6),
        "median_pair_margin": round(float(np.median(m)), 6),
        "positive_margin_rate": round(float(np.mean(m > 0)), 4),
        "mean_logit_diff_gap": round(float(np.mean(lg)), 6),
        "median_logit_diff_gap": round(float(np.median(lg)), 6),
    }


def _load_fhm_pairs(
    metadata_path: str,
    keys: list[str],
    probe_preds: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[str, str]]]:
    from gemma3_experiments.launch.residual.probe_validate import _load_fhm_confounder_metadata
    metadata = _load_fhm_confounder_metadata(Path(metadata_path))
    key_set = set(keys)
    from gemma3_experiments.launch.residual.analyze_fhm_confounder_features import _build_pairs

    rows = []
    for key in keys:
        pred_row = probe_preds.get(key)
        if pred_row is None:
            continue
        sample_id = key.split("::")[-1]
        gold_label = str(pred_row.get("gold_label", "not hateful"))
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_name": "Hateful_Memes_Dataset",
                "task_name": "hateful_memes_binary",
                "schema": {"primary_field": {"name": "label"}},
                "gold_fields": {"label": gold_label},
            }
        )

    pairs_by_source = {}
    for source in ["pseudo_img", "pseudo_text"]:
        pairs_by_source[source] = _build_pairs(
            rows,
            metadata, key_set, source
        )
    return pairs_by_source


def run_fhm_ablation(cfg: CausalAblationConfig, rng: np.random.Generator) -> dict[str, Any]:
    import scipy.sparse

    preds_path = Path(cfg.fhm_probe_predictions_path)
    feat_path = Path(cfg.fhm_feature_matrix_path)
    if not feat_path.exists() and preds_path.exists():
        fallback = preds_path.with_name("feature_matrix.npz")
        if fallback.exists():
            logger.info("Using fallback FHM feature matrix at %s", fallback)
            feat_path = fallback
    if not preds_path.exists() or not feat_path.exists():
        logger.warning(
            "FHM paths missing: preds=%s feats=%s. "
            "Re-run probe_validate with --save_feature_matrix true.",
            preds_path,
            feat_path,
        )
        return {}

    probe_preds = _load_probe_predictions(preds_path)
    keys, X = _load_feature_matrix(feat_path)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    # FHM binary label space (canonical order used throughout).
    FHM_CLASSES = ["hateful", "not hateful"]

    # Load fitted probe and (optionally) scaler saved alongside predictions.
    # NOTE: feature_matrix.npz contains *unscaled* activations; the probe was
    # trained with MaxAbsScaler applied.  Save probe.joblib as a sklearn Pipeline
    # (or save scaler.joblib separately) for post-ablation predictions to be valid.
    # Without a scaler, clf.predict / predict_proba results are approximate.
    probe_joblib_path = preds_path.with_name("probe.joblib")
    scaler_joblib_path = preds_path.with_name("scaler.joblib")
    le_joblib_path = preds_path.with_name("label_encoder.joblib")
    clf = None
    scaler = None
    le = None
    if probe_joblib_path.exists():
        import joblib
        clf = joblib.load(probe_joblib_path)
        logger.info("Loaded probe from %s", probe_joblib_path)
    if scaler_joblib_path.exists():
        import joblib
        scaler = joblib.load(scaler_joblib_path)
        logger.info("Loaded scaler from %s", scaler_joblib_path)
    if le_joblib_path.exists():
        import joblib
        le = joblib.load(le_joblib_path)
        logger.info("Loaded label encoder from %s (classes=%s)", le_joblib_path, le.classes_.tolist())

    def _predict_proba_scaled(X_raw: Any) -> "np.ndarray | None":
        """Apply optional scaler then predict_proba; align columns to FHM_CLASSES."""
        if clf is None or not hasattr(clf, "predict_proba"):
            return None
        try:
            X_in = scaler.transform(X_raw) if scaler is not None else X_raw
            raw = clf.predict_proba(X_in)
            # clf.classes_ are LabelEncoder integers; use le.classes_ for string names.
            src_classes = le.classes_.tolist() if le is not None else clf.classes_.tolist()
            return _align_proba_to_classes(raw, src_classes, FHM_CLASSES)
        except Exception as exc:
            logger.error("predict_proba failed: %s", exc)
            return None

    def _predict_labels_scaled(X_raw: Any) -> list[str]:
        if clf is None:
            return pred_labels
        try:
            X_in = scaler.transform(X_raw) if scaler is not None else X_raw
            enc = clf.predict(X_in)
            # enc contains LabelEncoder integers; invert via le when available.
            if le is not None:
                return le.inverse_transform(enc).tolist()
            return [str(clf.classes_.tolist()[p]) for p in enc]
        except Exception as exc:
            logger.error("clf.predict failed after ablation: %s", exc)
            return pred_labels

    # Build gold/pred arrays aligned to feature matrix.
    gold_labels: list[str] = []
    pred_labels: list[str] = []
    for key in keys:
        row = probe_preds.get(key)
        if row is None:
            gold_labels.append("not hateful")
            pred_labels.append("not hateful")
        else:
            gold_labels.append(row.get("gold_label", "not hateful"))
            pred_labels.append(row.get("probe_pred", "not hateful"))

    # ── Baseline probability margins from stored probe_proba (no clf needed) ──
    baseline_proba = _extract_proba_from_preds(keys, probe_preds, FHM_CLASSES)
    pairs_by_source = _load_fhm_pairs(cfg.fhm_confounder_metadata_path, keys, probe_preds)

    baseline_pair_margins: dict[str, Any] = {}
    if baseline_proba is not None:
        for source, pairs in pairs_by_source.items():
            baseline_pair_margins[source] = _compute_pair_margin_metrics(
                baseline_proba, FHM_CLASSES, pairs, key_to_idx
            )
        logger.info(
            "Baseline pair margins: %s",
            {s: m.get("mean_pair_margin") for s, m in baseline_pair_margins.items()},
        )
    else:
        logger.warning(
            "No stored probe_proba found in %s; baseline pair margins unavailable."
            " Re-run probe_validate to populate probe_proba in the predictions file.",
            preds_path,
        )

    # ── Main ablation: fhm_top_confounder_features ────────────────────────────
    features_to_ablate = cfg.fhm_top_confounder_features[: cfg.top_k_ablate]
    X_ablated = _ablate_features(X, features_to_ablate, cfg.ablation_method, rng)

    y_pred = _predict_labels_scaled(X_ablated)
    raw_metrics = _per_class_metrics(gold_labels, y_pred)

    ablated_proba = _predict_proba_scaled(X_ablated)
    ablated_pair_margins: dict[str, Any] = {}
    pair_margin_delta: dict[str, Any] = {}
    if ablated_proba is not None:
        for source, pairs in pairs_by_source.items():
            ablated_pair_margins[source] = _compute_pair_margin_metrics(
                ablated_proba, FHM_CLASSES, pairs, key_to_idx
            )
        for source in pairs_by_source:
            bm = baseline_pair_margins.get(source, {})
            am = ablated_pair_margins.get(source, {})
            if bm and am:
                pair_margin_delta[source] = {
                    "mean_pair_margin_delta": round(
                        am.get("mean_pair_margin", 0.0) - bm.get("mean_pair_margin", 0.0), 6
                    ),
                    "positive_margin_rate_delta": round(
                        am.get("positive_margin_rate", 0.0) - bm.get("positive_margin_rate", 0.0), 4
                    ),
                    "mean_logit_diff_gap_delta": round(
                        am.get("mean_logit_diff_gap", 0.0) - bm.get("mean_logit_diff_gap", 0.0), 6
                    ),
                }
        logger.info(
            "Main ablation pair margin delta: %s",
            {s: d.get("mean_pair_margin_delta") for s, d in pair_margin_delta.items()},
        )
    elif clf is not None:
        logger.warning(
            "predict_proba unavailable; post-ablation pair margins require a probe "
            "with probability output saved to %s.",
            probe_joblib_path,
        )

    # ── Legacy binary pair metrics (both_correct / separated) ─────────────────
    pair_metrics: dict[str, Any] = {}
    for source, pairs in pairs_by_source.items():
        both_correct = 0
        separated = 0
        n = len(pairs)
        for h_key, b_key in pairs:
            h_idx = key_to_idx.get(h_key)
            b_idx = key_to_idx.get(b_key)
            if h_idx is None or b_idx is None:
                continue
            h_pred = y_pred[h_idx]
            b_pred = y_pred[b_idx]
            h_gold = gold_labels[h_idx]
            b_gold = gold_labels[b_idx]
            if h_pred == h_gold and b_pred == b_gold:
                both_correct += 1
            if h_pred != b_pred:
                separated += 1
        pair_metrics[source] = {
            "n_pairs": n,
            "both_correct": both_correct,
            "both_correct_rate": both_correct / max(n, 1),
            "separated": separated,
            "separated_rate": separated / max(n, 1),
        }

    # ── Named feature-group ablations ─────────────────────────────────────────
    # Each group is ablated independently; pair-margin delta vs baseline is reported.
    feature_groups = (
        _parse_class_feature_entries(cfg.fhm_feature_groups_entries)
        if cfg.fhm_feature_groups_entries
        else {}
    )
    group_ablation_results: dict[str, Any] = {}
    for group_name, group_feats in feature_groups.items():
        X_g = _ablate_features(X, group_feats, cfg.ablation_method, rng)
        group_proba = _predict_proba_scaled(X_g)
        group_pair_margins: dict[str, Any] = {}
        group_delta: dict[str, Any] = {}
        if group_proba is not None:
            for source, pairs in pairs_by_source.items():
                group_pair_margins[source] = _compute_pair_margin_metrics(
                    group_proba, FHM_CLASSES, pairs, key_to_idx
                )
            for source in pairs_by_source:
                bm = baseline_pair_margins.get(source, {})
                gm = group_pair_margins.get(source, {})
                if bm and gm:
                    group_delta[source] = {
                        "mean_pair_margin_delta": round(
                            gm.get("mean_pair_margin", 0.0) - bm.get("mean_pair_margin", 0.0), 6
                        ),
                        "positive_margin_rate_delta": round(
                            gm.get("positive_margin_rate", 0.0) - bm.get("positive_margin_rate", 0.0), 4
                        ),
                        "mean_logit_diff_gap_delta": round(
                            gm.get("mean_logit_diff_gap", 0.0) - bm.get("mean_logit_diff_gap", 0.0), 6
                        ),
                    }
        group_ablation_results[group_name] = {
            "feature_indices": group_feats,
            "n_features": len(group_feats),
            "pair_margins": group_pair_margins,
            "pair_margin_delta": group_delta,
        }
        logger.info(
            "Group ablation '%s' (%d feats): pair_margins=%s delta=%s",
            group_name,
            len(group_feats),
            {s: m.get("mean_pair_margin") for s, m in group_pair_margins.items()},
            {s: d.get("mean_pair_margin_delta") for s, d in group_delta.items()},
        )

    return {
        "ablation_method": cfg.ablation_method,
        "n_features_ablated": len(features_to_ablate),
        "feature_indices": features_to_ablate,
        "raw_metrics": raw_metrics,
        "pair_metrics": pair_metrics,
        "baseline_pair_margins": baseline_pair_margins,
        "ablated_pair_margins": ablated_pair_margins,
        "pair_margin_delta": pair_margin_delta,
        "group_ablations": group_ablation_results,
    }


def run_mmhs_ablation(cfg: CausalAblationConfig, rng: np.random.Generator) -> dict[str, Any]:
    import scipy.sparse

    preds_path = Path(cfg.mmhs_probe_predictions_path)
    feat_path = Path(cfg.mmhs_feature_matrix_path)
    if not feat_path.exists() and preds_path.exists():
        fallback = preds_path.with_name("feature_matrix.npz")
        if fallback.exists():
            logger.info("Using fallback MMHS feature matrix at %s", fallback)
            feat_path = fallback
    if not preds_path.exists() or not feat_path.exists():
        logger.warning(
            "MMHS paths missing: preds=%s feats=%s. "
            "Re-run probe_validate with --save_feature_matrix true.",
            preds_path,
            feat_path,
        )
        return {}

    probe_preds = _load_probe_predictions(preds_path)
    keys, X = _load_feature_matrix(feat_path)
    key_to_idx = {k: i for i, k in enumerate(keys)}

    gold_labels = []
    pred_labels = []
    for key in keys:
        row = probe_preds.get(key)
        if row is None:
            gold_labels.append("NotHate")
            pred_labels.append("NotHate")
        else:
            gold_labels.append(row.get("gold_label", "NotHate"))
            pred_labels.append(row.get("probe_pred", "NotHate"))

    features_by_class = _normalized_mmhs_top_features(cfg)
    if not features_by_class:
        logger.warning("No MMHS class features configured; skipping MMHS ablation.")
        return {}

    results = {}
    for cls, feats in features_by_class.items():
        top_feats = feats[: cfg.top_k_ablate]
        X_ablated = _ablate_features(X, top_feats, cfg.ablation_method, rng)
        y_pred = pred_labels
        try:
            import joblib
            probe_joblib_path = preds_path.with_name("probe.joblib")
            if probe_joblib_path.exists():
                clf = joblib.load(probe_joblib_path)
                y_pred_enc = clf.predict(X_ablated)
                classes = clf.classes_.tolist()
                y_pred = [classes[p] for p in y_pred_enc]
        except Exception as e:
            logger.error("MMHS predict failed for class %s: %s", cls, e)

        metrics = _per_class_metrics(gold_labels, y_pred)
        cls_metrics = metrics.get("per_class", {}).get(cls, {})
        results[cls] = {
            "feature_indices": top_feats,
            "n_features": len(top_feats),
            "per_class": cls_metrics,
            "overall_f1": metrics.get("macro_f1"),
        }
    return results


def run(cfg: CausalAblationConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    results = {
        "config": {
            "ablation_method": cfg.ablation_method,
            "top_k_ablate": cfg.top_k_ablate,
        },
        "fhm": {},
        "mmhs": {},
    }

    if cfg.fhm_probe_predictions_path and cfg.fhm_feature_matrix_path:
        logger.info("Running FHM causal ablation")
        results["fhm"] = run_fhm_ablation(cfg, rng)

    if cfg.mmhs_probe_predictions_path and cfg.mmhs_feature_matrix_path:
        logger.info("Running MMHS causal ablation")
        results["mmhs"] = run_mmhs_ablation(cfg, rng)

    out_path = output_dir / "causal_ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote causal ablation results to %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(CausalAblationConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()