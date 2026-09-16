

import json
import logging
import numpy as np
import scipy.sparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)


@dataclass
class FeatureCausalCardConfig:
    # Key features to build causal cards for
    feature_indices: list[int] = field(default_factory=lambda: [
        14008, 2261, 1542, 19691, 4163,
        8253, 19940, 5743, 18499, 3753, 15881,
    ])

    # Input data sources
    feature_cards_path: str = ""                 # existing feature_cards.json
    confounder_audit_path: str = ""              # confounder_feature_top_features.jsonl
    probe_top_features_path: str = ""            # probe_results_harm_binary/top_features.json
    logit_alignment_path: str = ""               # feature_logit_alignment.json
    causal_ablation_path: str = ""               # causal_ablation_results.json
    probe_predictions_path: str = ""             # probe_val_predictions.jsonl
    feature_matrix_path: str = ""               # feature_matrix.npz (unscaled)
    feature_matrix_keys_path: str = ""          # feature_matrix_keys.npy
    probe_joblib_path: str = ""                  # probe.joblib (sklearn clf)
    scaler_joblib_path: str = ""                 # scaler.joblib (MaxAbsScaler)
    label_encoder_joblib_path: str = ""          # label_encoder.joblib
    feature_descriptions_path: str = ""          # feature_descriptions.json
    fhm_confounder_metadata_path: str = ""       # ivl_plus_gemini_captions_complete.json

    # Computation settings
    top_k_examples: int = 5       # FP/FN examples to include per feature
    pair_patch_n: int = 30        # max pairs to patch per source (sampling if more)
    ablation_method: str = "mean" # zero | mean

    output_dir: str = ""


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_feature_matrix(npz_path: Path, keys_path: Path | None) -> tuple[list[str], Any]:
    payload = np.load(str(npz_path), allow_pickle=True)
    if "keys" in payload and "data" in payload:
        # causal_feature_ablation.py format: keys array embedded in the npz
        keys = [str(k) for k in payload["keys"]]
        X = scipy.sparse.csr_matrix(
            (payload["data"], payload["indices"], payload["indptr"]),
            shape=tuple(payload["shape"]),
        )
    elif "keys" in payload and "X" in payload:
        # create_feature_matrix_npz.py format: np.savez_compressed(keys=..., X=sparse)
        # X is stored as a pickled object array; unwrap with [()]
        keys = [str(k) for k in payload["keys"]]
        X = payload["X"][()]
        if not scipy.sparse.issparse(X):
            X = scipy.sparse.csr_matrix(X)
    elif "keys" in payload:
        # Hybrid: keys embedded but matrix saved via scipy.sparse.save_npz
        keys = [str(k) for k in payload["keys"]]
        X = scipy.sparse.load_npz(str(npz_path))
    else:
        # scipy.sparse.save_npz format (probe_fhm_crosscoder.py).
        # Keys are in a sibling .npy file; matrix must be loaded with load_npz.
        assert keys_path is not None and keys_path.exists(), (
            f"feature_matrix.npz has no embedded 'keys' array; "
            f"supply --feature_matrix_keys_path pointing to the sibling .npy file"
        )
        keys = [str(k) for k in np.load(str(keys_path), allow_pickle=True)]
        X = scipy.sparse.load_npz(str(npz_path))
    return keys, X


def _load_probe(joblib_path: Path, scaler_path: Path | None, le_path: Path | None):
    import joblib
    clf = joblib.load(str(joblib_path))
    scaler = joblib.load(str(scaler_path)) if scaler_path and scaler_path.exists() else None
    le = joblib.load(str(le_path)) if le_path and le_path.exists() else None
    return clf, scaler, le


# ─────────────────────────────────────────────────────────────
# Pseudo-pair building
# ─────────────────────────────────────────────────────────────

def _build_pseudo_pairs(
    fhm_metadata_path: Path,
    keys: list[str],
    preds_by_key: dict[str, dict],
) -> dict[str, list[tuple[str, str]]]:
    """Return {source: [(hateful_key, benign_key)]} for both pseudo_img and pseudo_text."""
    try:
        from gemma3_experiments.launch.residual.analyze_fhm_confounder_features import _build_pairs
        from gemma3_experiments.launch.residual.probe_validate import _load_fhm_confounder_metadata, _row_gold_label
    except ImportError:
        logger.warning("Could not import pair-building helpers; skipping pair metrics.")
        return {}

    metadata = _load_fhm_confounder_metadata(fhm_metadata_path)
    key_set = set(keys)
    rows = []
    for key in keys:
        pred = preds_by_key.get(key, {})
        gold = pred.get("gold_label", "not hateful")
        sample_id = key.split("::")[-1]
        rows.append({
            "sample_id": sample_id,
            "dataset_name": "Hateful_Memes_Dataset",
            "task_name": "hateful_memes_binary",
            "schema": {"primary_field": {"name": "label"}},
            "gold_fields": {"label": gold},
        })
    pairs: dict[str, list[tuple[str, str]]] = {}
    for src in ("pseudo_img", "pseudo_text"):
        pairs[src] = _build_pairs(rows, metadata, key_set, src)
    return pairs


# ─────────────────────────────────────────────────────────────
# Predict helpers
# ─────────────────────────────────────────────────────────────

def _predict_proba(X_in: Any, clf, scaler, le, classes: list[str]) -> np.ndarray:
    """Return (n, n_classes) float32 array aligned to `classes`."""
    X_s = scaler.transform(X_in) if scaler is not None else X_in
    # sklearn MLP needs dense
    if hasattr(X_s, "toarray"):
        X_s = X_s.toarray()
    raw = clf.predict_proba(X_s).astype(np.float32)
    src = le.classes_.tolist() if le is not None else clf.classes_.tolist()
    aligned = np.zeros((raw.shape[0], len(classes)), dtype=np.float32)
    for j, c in enumerate(classes):
        if c in src:
            aligned[:, j] = raw[:, src.index(c)]
    return aligned


def _pair_margins(
    proba: np.ndarray,
    pairs: list[tuple[str, str]],
    key_to_idx: dict[str, int],
    pos_col: int,
) -> dict[str, float]:
    margins = []
    for hk, bk in pairs:
        hi = key_to_idx.get(hk)
        bi = key_to_idx.get(bk)
        if hi is None or bi is None:
            continue
        margins.append(float(proba[hi, pos_col] - proba[bi, pos_col]))
    if not margins:
        return {}
    m = np.array(margins, dtype=np.float64)
    return {
        "n_pairs": len(m),
        "mean_margin": round(float(np.mean(m)), 5),
        "positive_margin_rate": round(float(np.mean(m > 0)), 4),
        "median_margin": round(float(np.median(m)), 5),
    }


# ─────────────────────────────────────────────────────────────
# Per-feature ablation (offline)
# ─────────────────────────────────────────────────────────────

def _ablate_one(
    X: Any,
    feat_col: int,
    method: str,
    train_mean: float,
) -> Any:
    X_abl = X.copy().tolil()
    if method == "zero":
        X_abl[:, feat_col] = 0
    elif method == "mean":
        col = X_abl[:, feat_col].toarray().ravel() - train_mean
        X_abl[:, feat_col] = col[:, np.newaxis]
    X_abl = X_abl.tocsr()
    return X_abl


# ─────────────────────────────────────────────────────────────
# Patching effect (causal sufficiency)
# ─────────────────────────────────────────────────────────────

def _patching_effect(
    X: Any,
    feat_col: int,
    pairs: list[tuple[str, str]],
    key_to_idx: dict[str, int],
    clf,
    scaler,
    le,
    classes: list[str],
    pos_col: int,
    rng: np.random.Generator,
    max_pairs: int = 30,
) -> dict[str, Any]:
    """For each pair, swap feature value between members and measure P(hateful) change."""
    filtered = [(hk, bk) for hk, bk in pairs
                if hk in key_to_idx and bk in key_to_idx]
    if not filtered:
        return {}
    if len(filtered) > max_pairs:
        idx = rng.choice(len(filtered), max_pairs, replace=False)
        filtered = [filtered[i] for i in idx]

    # Baseline probabilities for all relevant rows
    all_idx = sorted(set(
        [key_to_idx[k] for k in [hk for hk, _ in filtered] + [bk for _, bk in filtered]]
    ))
    X_sub = X[all_idx]
    baseline_proba = _predict_proba(X_sub, clf, scaler, le, classes)
    # Row index in X_sub → probability
    local_map = {global_idx: local_i for local_i, global_idx in enumerate(all_idx)}

    patch_b2h_deltas = []   # benign→hateful patch: should reduce hateful P
    patch_h2b_deltas = []   # hateful→benign patch: should increase benign P

    for hk, bk in filtered:
        hi = key_to_idx[hk]
        bi = key_to_idx[bk]
        h_val = float(X[hi, feat_col])
        b_val = float(X[bi, feat_col])

        # Patch benign→hateful: replace hateful's feature with benign's value
        X_patch = X[[hi]].copy().tolil()
        X_patch[0, feat_col] = b_val
        p_h_patched = float(_predict_proba(X_patch.tocsr(), clf, scaler, le, classes)[0, pos_col])
        p_h_base = float(baseline_proba[local_map[hi], pos_col])
        patch_b2h_deltas.append(p_h_patched - p_h_base)

        # Patch hateful→benign: replace benign's feature with hateful's value
        X_patch = X[[bi]].copy().tolil()
        X_patch[0, feat_col] = h_val
        p_b_patched = float(_predict_proba(X_patch.tocsr(), clf, scaler, le, classes)[0, pos_col])
        p_b_base = float(baseline_proba[local_map[bi], pos_col])
        patch_h2b_deltas.append(p_b_patched - p_b_base)

    b2h = np.array(patch_b2h_deltas, dtype=np.float64)
    h2b = np.array(patch_h2b_deltas, dtype=np.float64)
    return {
        "n_pairs": len(filtered),
        # b2h: patching benign value into hateful member → should make prediction LESS hateful
        "patch_b2h_mean_delta": round(float(np.mean(b2h)), 5),
        "patch_b2h_negative_rate": round(float(np.mean(b2h < -0.01)), 4),  # % of pairs where hateful P drops
        # h2b: patching hateful value into benign member → should make prediction MORE hateful
        "patch_h2b_mean_delta": round(float(np.mean(h2b)), 5),
        "patch_h2b_positive_rate": round(float(np.mean(h2b > 0.01)), 4),  # % of pairs where benign P rises
        # Combined causal strength: feature is causal if both directions work
        "causal_score": round(float(-np.mean(b2h) + np.mean(h2b)) / 2, 5),
    }


# ─────────────────────────────────────────────────────────────
# FP/FN examples
# ─────────────────────────────────────────────────────────────

def _fp_fn_examples(
    X: Any,
    feat_col: int,
    keys: list[str],
    preds_by_key: dict[str, dict],
    top_k: int = 5,
) -> dict[str, list[dict]]:
    acts = np.asarray(X[:, feat_col].todense()).ravel()
    key_to_idx = {k: i for i, k in enumerate(keys)}

    fp, fn = [], []
    for key, pred in preds_by_key.items():
        idx = key_to_idx.get(key)
        if idx is None:
            continue
        if pred.get("probe_pred") == "hateful" and pred.get("gold_label") == "not hateful":
            fp.append((acts[idx], key, pred))
        elif pred.get("probe_pred") == "not hateful" and pred.get("gold_label") == "hateful":
            fn.append((acts[idx], key, pred))

    def _top(lst):
        lst.sort(key=lambda x: -x[0])
        return [
            {
                "feature_activation": round(float(a), 4),
                "sample_key": k,
                "meme_text": p.get("meme_text", ""),
                "gold_label": p.get("gold_label", ""),
                "probe_proba": p.get("probe_proba", {}),
            }
            for a, k, p in lst[:top_k]
        ]

    return {"false_positives": _top(fp), "false_negatives": _top(fn)}


# ─────────────────────────────────────────────────────────────
# Main assembly
# ─────────────────────────────────────────────────────────────

def run(cfg: FeatureCausalCardConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # ── Load data sources ──────────────────────────────────────────────────────
    logger.info("Loading data sources …")

    # 1. Existing feature cards (exemplars + NL descriptions + probe_sources)
    cards_by_id: dict[int, dict] = {}
    if cfg.feature_cards_path and Path(cfg.feature_cards_path).exists():
        raw = json.loads(Path(cfg.feature_cards_path).read_text())
        for c in raw.get("features", []):
            cards_by_id[int(c["feature_idx"])] = c
        logger.info("Loaded %d existing cards", len(cards_by_id))

    # 2. NL descriptions
    descriptions: dict[int, str] = {}
    if cfg.feature_descriptions_path and Path(cfg.feature_descriptions_path).exists():
        raw = json.loads(Path(cfg.feature_descriptions_path).read_text())
        for feat_key, entry in raw.items():
            try:
                if isinstance(entry, str):
                    descriptions[int(feat_key)] = entry
                else:
                    descriptions[int(feat_key)] = entry.get("description", "")
            except (ValueError, TypeError):
                pass
        logger.info("Loaded %d NL descriptions", len(descriptions))

    # 3. Confounder audit
    audit_by_feat: dict[int, list[dict]] = {}
    if cfg.confounder_audit_path and Path(cfg.confounder_audit_path).exists():
        for r in _load_jsonl(Path(cfg.confounder_audit_path)):
            fid = int(r["feature_idx"])
            audit_by_feat.setdefault(fid, []).append(r)
        logger.info("Loaded confounder audit: %d features", len(audit_by_feat))

    # 4. Probe top features
    probe_by_feat: dict[int, list[dict]] = {}
    if cfg.probe_top_features_path and Path(cfg.probe_top_features_path).exists():
        raw = json.loads(Path(cfg.probe_top_features_path).read_text())
        for task_key, class_dict in raw.items():
            if not isinstance(class_dict, dict):
                continue
            for class_name, feats in class_dict.items():
                for f in feats:
                    fid = int(f["feature_idx"])
                    probe_by_feat.setdefault(fid, []).append({
                        "task_key": task_key,
                        "class_name": class_name,
                        "weight": f.get("weight", 0.0),
                        "abs_weight": abs(f.get("weight", 0.0)),
                    })
        logger.info("Loaded probe weights: %d features", len(probe_by_feat))

    # 5. Logit alignment
    logit_by_feat: dict[int, list[dict]] = {}
    if cfg.logit_alignment_path and Path(cfg.logit_alignment_path).exists():
        raw = json.loads(Path(cfg.logit_alignment_path).read_text())
        for task_key, class_dict in raw.items():
            for class_name, rows in class_dict.items():
                for r in rows:
                    fid = int(r["feature_idx"])
                    logit_by_feat.setdefault(fid, []).append({
                        "task_key": task_key,
                        "class_name": class_name,
                        "probe_weight": r.get("probe_weight", 0.0),
                        "static_logit_effect": r.get("static_logit_effect", 0.0),
                        "quadrant": r.get("quadrant", "unknown"),
                    })
        logger.info("Loaded logit alignment: %d features", len(logit_by_feat))

    # 6. Causal ablation (group-level only)
    ablation_groups: dict[str, dict] = {}
    if cfg.causal_ablation_path and Path(cfg.causal_ablation_path).exists():
        ca = json.loads(Path(cfg.causal_ablation_path).read_text())
        fhm = ca.get("fhm", {})
        for grp, v in fhm.get("group_ablations", {}).items():
            ablation_groups[grp] = {
                "features": v.get("feature_indices", []),
                "pair_margin_delta": v.get("pair_margin_delta", {}),
            }
        logger.info("Loaded causal ablation groups: %s", list(ablation_groups.keys()))

    # 7. Feature matrix + probe (for offline ablation, patching, FP/FN)
    X = keys = clf = scaler = le = None
    key_to_idx: dict[str, int] = {}
    preds_by_key: dict[str, dict] = {}
    pairs_by_source: dict[str, list] = {}

    if cfg.feature_matrix_path and Path(cfg.feature_matrix_path).exists():
        keys_p = Path(cfg.feature_matrix_keys_path) if cfg.feature_matrix_keys_path else None
        keys, X = _load_feature_matrix(Path(cfg.feature_matrix_path), keys_p)
        key_to_idx = {k: i for i, k in enumerate(keys)}
        logger.info("Loaded feature matrix: %s", X.shape)

    if cfg.probe_joblib_path and Path(cfg.probe_joblib_path).exists():
        clf, scaler, le = _load_probe(
            Path(cfg.probe_joblib_path),
            Path(cfg.scaler_joblib_path) if cfg.scaler_joblib_path else None,
            Path(cfg.label_encoder_joblib_path) if cfg.label_encoder_joblib_path else None,
        )
        logger.info("Loaded probe clf=%s", type(clf).__name__)

    if cfg.probe_predictions_path and Path(cfg.probe_predictions_path).exists():
        for r in _load_jsonl(Path(cfg.probe_predictions_path)):
            k = r.get("sample_key") or r.get("task_key", "") + "::" + str(r.get("sample_id", ""))
            preds_by_key[k] = r
        logger.info("Loaded %d probe predictions", len(preds_by_key))

    if cfg.fhm_confounder_metadata_path and Path(cfg.fhm_confounder_metadata_path).exists() and keys:
        try:
            pairs_by_source = _build_pseudo_pairs(
                Path(cfg.fhm_confounder_metadata_path), keys, preds_by_key
            )
            for src, p in pairs_by_source.items():
                logger.info("Pseudo pairs: %s=%d", src, len(p))
        except Exception as exc:
            logger.warning("Pair building failed: %s", exc)

    # Precompute training-set mean activations for mean-ablation
    train_means: dict[int, float] = {}
    if X is not None:
        for feat_col in cfg.feature_indices:
            if feat_col < X.shape[1]:
                train_means[feat_col] = float(X[:, feat_col].mean())

    # FHM classes
    FHM_CLASSES = ["hateful", "not hateful"]
    pos_col = 0  # hateful is index 0 in sorted classes

    # ── Build causal card for each key feature ──────────────────────────────────
    causal_cards = []

    for feat_idx in cfg.feature_indices:
        logger.info("Building causal card for f%d …", feat_idx)
        card: dict[str, Any] = {"feature_idx": feat_idx}

        # 1. Top examples (from existing cards)
        existing = cards_by_id.get(feat_idx, {})
        card["top_examples"] = existing.get("top_examples", [])
        card["sources"] = existing.get("sources", [])

        # 2. NL description
        card["nl_description"] = descriptions.get(feat_idx, "")

        # 3. Confounder paired deltas
        audit_entries = audit_by_feat.get(feat_idx, [])
        confounder_summary: dict[str, dict] = {}
        for r in audit_entries:
            src = r.get("source", "unknown")
            tok = r.get("token_selection", "unknown")
            key = f"{tok}/{src}"
            confounder_summary[key] = {
                "rank": r.get("rank"),
                "score": round(float(r.get("score", 0)), 5),
                "sign_consistency": round(float(r.get("sign_consistency", 0)), 4),
                "mean_delta": round(float(r.get("mean_delta", 0)), 3),
                "direction": r.get("direction", ""),
            }
        card["confounder_pairs"] = confounder_summary

        # 4. Probe weights (top entries by abs_weight)
        probe_entries = probe_by_feat.get(feat_idx, [])
        probe_entries.sort(key=lambda x: -abs(x.get("weight", 0)))
        card["probe_weights"] = probe_entries[:10]

        # 5. Logit attribution
        logit_entries = logit_by_feat.get(feat_idx, [])
        logit_entries.sort(key=lambda x: -abs(x.get("static_logit_effect", 0)))
        card["logit_attribution"] = logit_entries[:10]

        # 6. Causal ablation (group membership + group deltas)
        feature_group_membership = []
        for grp, gv in ablation_groups.items():
            if feat_idx in gv.get("features", []):
                feature_group_membership.append({
                    "group": grp,
                    "pair_margin_delta": gv.get("pair_margin_delta", {}),
                })
        card["ablation_group_effect"] = feature_group_membership

        # 6b. Per-feature offline ablation (if matrix + probe available)
        per_feat_ablation: dict[str, Any] = {}
        if X is not None and clf is not None and feat_idx < X.shape[1] and pairs_by_source:
            baseline_proba = _predict_proba(X, clf, scaler, le, FHM_CLASSES)
            # Ablate this feature
            X_abl = _ablate_one(X, feat_idx, cfg.ablation_method, train_means.get(feat_idx, 0.0))
            ablated_proba = _predict_proba(X_abl, clf, scaler, le, FHM_CLASSES)
            for src, pairs in pairs_by_source.items():
                baseline_m = _pair_margins(baseline_proba, pairs, key_to_idx, pos_col)
                ablated_m = _pair_margins(ablated_proba, pairs, key_to_idx, pos_col)
                if baseline_m and ablated_m:
                    per_feat_ablation[src] = {
                        "baseline_mean_margin": baseline_m["mean_margin"],
                        "ablated_mean_margin": ablated_m["mean_margin"],
                        "delta": round(ablated_m["mean_margin"] - baseline_m["mean_margin"], 5),
                        "pos_rate_baseline": baseline_m["positive_margin_rate"],
                        "pos_rate_ablated": ablated_m["positive_margin_rate"],
                        "pos_rate_delta": round(
                            ablated_m["positive_margin_rate"] - baseline_m["positive_margin_rate"], 4
                        ),
                    }
        card["ablation_per_feature"] = per_feat_ablation

        # 7. Patching effect (causal sufficiency)
        patching: dict[str, Any] = {}
        if X is not None and clf is not None and feat_idx < X.shape[1] and pairs_by_source:
            for src, pairs in pairs_by_source.items():
                patching[src] = _patching_effect(
                    X, feat_idx, pairs, key_to_idx,
                    clf, scaler, le, FHM_CLASSES, pos_col,
                    rng, cfg.pair_patch_n,
                )
        card["patching_effect"] = patching

        # 8. FP/FN examples
        if X is not None and feat_idx < X.shape[1] and preds_by_key:
            card["fp_fn_examples"] = _fp_fn_examples(
                X, feat_idx, keys, preds_by_key, cfg.top_k_examples
            )
        else:
            card["fp_fn_examples"] = {"false_positives": [], "false_negatives": []}

        causal_cards.append(card)

    # ── Write outputs ────────────────────────────────────────────────────────────
    (output_dir / "feature_causal_cards.json").write_text(
        json.dumps({"features": causal_cards}, indent=2)
    )

    # Human-readable text card per feature
    for card in causal_cards:
        _write_text_card(card, output_dir)

    # Summary ranked by causal score
    summary = []
    for card in causal_cards:
        best_patch_score = max(
            (v.get("causal_score", 0) for v in card["patching_effect"].values()),
            default=0.0,
        )
        best_ablation_delta = min(
            (v.get("delta", 0) for v in card["ablation_per_feature"].values()),
            default=0.0,
        )
        best_logit_effect = max(
            (abs(r.get("static_logit_effect", 0)) for r in card.get("logit_attribution", [])),
            default=0.0,
        )
        best_confounder_rate = max(
            (v.get("sign_consistency", 0) for v in card.get("confounder_pairs", {}).values()),
            default=0.0,
        )
        summary.append({
            "feature_idx": card["feature_idx"],
            "nl_description_snippet": (card.get("nl_description") or "")[:100],
            "best_patch_causal_score": round(best_patch_score, 4),
            "best_ablation_margin_delta": round(best_ablation_delta, 5),
            "best_logit_effect": round(best_logit_effect, 4),
            "best_confounder_sign_consistency": round(best_confounder_rate, 4),
            "n_probe_tasks": len(set(e["task_key"] for e in card.get("probe_weights", []))),
        })
    summary.sort(key=lambda x: -x["best_patch_causal_score"])
    (output_dir / "feature_causal_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("Wrote %d causal cards to %s", len(causal_cards), output_dir)


def _write_text_card(card: dict, output_dir: Path) -> None:
    """Write a human-readable text card for one feature."""
    fid = card["feature_idx"]
    lines = [f"{'='*60}", f"Feature f{fid} — Causal Card", f"{'='*60}", ""]

    desc = card.get("nl_description", "")
    if desc:
        lines += [f"SEMANTIC DESCRIPTION:\n  {desc[:300]}", ""]

    # Probe weights (top 3)
    pw = card.get("probe_weights", [])
    if pw:
        lines.append("PROBE WEIGHTS (top 3):")
        for e in pw[:3]:
            lines.append(f"  [{e['task_key']}] class={e['class_name']}  w={e['weight']:.4f}")
        lines.append("")

    # Logit attribution (top 3)
    la = card.get("logit_attribution", [])
    if la:
        lines.append("MODEL LOGIT ATTRIBUTION (direct, static):")
        for e in la[:3]:
            lines.append(f"  [{e['task_key']}] class={e['class_name']}  "
                         f"logit_effect={e['static_logit_effect']:.5f}  quadrant={e['quadrant']}")
        lines.append("")

    # Confounder pairs
    cp = card.get("confounder_pairs", {})
    if cp:
        lines.append("CONFOUNDER PAIRED DELTAS (sign_consistency):")
        for key, v in sorted(cp.items()):
            lines.append(f"  [{key}]  rank={v['rank']}  score={v['score']:.4f}  "
                         f"sign_cons={v['sign_consistency']:.3f}  dir={v['direction']}")
        lines.append("")

    # Per-feature ablation
    ab = card.get("ablation_per_feature", {})
    if ab:
        lines.append(f"PER-FEATURE ABLATION EFFECT (method=mean):")
        for src, v in ab.items():
            lines.append(f"  [{src}]  margin_delta={v['delta']:+.5f}  "
                         f"pos_rate_delta={v['pos_rate_delta']:+.4f}")
        lines.append("")

    # Patching effect
    pe = card.get("patching_effect", {})
    if pe:
        lines.append("PATCHING EFFECT (causal sufficiency):")
        for src, v in pe.items():
            if not v:
                continue
            lines.append(f"  [{src}]  n={v['n_pairs']}  "
                         f"causal_score={v['causal_score']:+.4f}  "
                         f"b2h_mean_delta={v['patch_b2h_mean_delta']:+.5f} (neg=good)  "
                         f"h2b_mean_delta={v['patch_h2b_mean_delta']:+.5f} (pos=good)")
        lines.append("")

    # Ablation group membership
    ag = card.get("ablation_group_effect", [])
    if ag:
        lines.append("GROUP ABLATION MEMBERSHIP:")
        for e in ag:
            pmd = e.get("pair_margin_delta", {})
            for src, m in pmd.items():
                lines.append(f"  group={e['group']}  [{src}]  "
                             f"margin_delta={m.get('mean_pair_margin_delta', 0):+.5f}")
        lines.append("")

    # FP/FN examples
    fpfn = card.get("fp_fn_examples", {})
    fps = fpfn.get("false_positives", [])
    fns = fpfn.get("false_negatives", [])
    if fps:
        lines.append(f"FALSE POSITIVES (pred=hateful, gold=not_hateful, top by activation):")
        for ex in fps[:3]:
            lines.append(f"  act={ex['feature_activation']:.3f}  text={ex['meme_text'][:80]!r}")
        lines.append("")
    if fns:
        lines.append(f"FALSE NEGATIVES (pred=not_hateful, gold=hateful, top by activation):")
        for ex in fns[:3]:
            lines.append(f"  act={ex['feature_activation']:.3f}  text={ex['meme_text'][:80]!r}")
        lines.append("")

    # Top examples
    exs = card.get("top_examples", [])
    if exs:
        lines.append(f"TOP ACTIVATING EXAMPLES (n={len(exs)}):")
        for ex in exs[:5]:
            lines.append(f"  act={ex.get('activation',0):.3f}  "
                         f"gold={ex.get('gold_label','')}  "
                         f"text={ex.get('meme_text','')[:60]!r}")
        lines.append("")

    (output_dir / f"causal_card_{fid}.txt").write_text("\n".join(lines))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FeatureCausalCardConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
