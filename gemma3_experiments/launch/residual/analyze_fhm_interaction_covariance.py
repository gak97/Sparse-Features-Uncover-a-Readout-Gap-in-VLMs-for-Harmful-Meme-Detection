

import json
import logging
import numpy as np
import scipy.sparse
from dataclasses import dataclass, field
from pathlib import Path

from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)


@dataclass
class FHMInteractionCovarianceConfig:
    # Pre-built image-token and prompt-token feature matrices
    image_matrix_path: str = ""          # feature_matrix_image.npz
    prompt_matrix_path: str = ""         # feature_matrix_prompt.npz
    image_matrix_keys_path: str = ""     # keys .npy (if separate)
    prompt_matrix_keys_path: str = ""

    # Confounder audit — restricts analysis to top-K confounder features
    audit_path: str = ""
    bilinear_top_k: int = 128
    audit_token_selection: str = "prompt_and_image"

    # Probe predictions OR cache manifest — exactly one must be supplied for
    # gold labels.  Use probe_predictions_path for the validation split (469
    # samples); use cache_dir for the full training split (7938 samples, more
    # stable covariance estimates).
    probe_predictions_path: str = ""     # probe_val_predictions.jsonl
    cache_dir: str = ""                  # residual SAE cache dir (reads manifest)

    # NL descriptions for feature labelling
    feature_descriptions_path: str = ""

    # Analysis settings
    top_directions: int = 8              # top ΔΣ_IP SVD directions
    top_k_features: int = 10            # top features per direction
    top_k_examples: int = 8            # examples per quadrant
    threshold_pct: float = 75.0         # percentile for "fires" threshold

    output_dir: str = ""


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_matrix(npz_path: Path, keys_path: Path | None) -> tuple[list[str], np.ndarray]:
    """Load feature matrix, returning (keys, dense_array)."""
    payload = np.load(str(npz_path), allow_pickle=True)
    if "keys" in payload and "X" in payload:
        keys = [str(k) for k in payload["keys"]]
        X = payload["X"][()]
    elif "keys" in payload and "data" in payload:
        keys = [str(k) for k in payload["keys"]]
        X = scipy.sparse.csr_matrix(
            (payload["data"], payload["indices"], payload["indptr"]),
            shape=tuple(payload["shape"]),
        )
    else:
        assert keys_path is not None and keys_path.exists()
        keys = [str(k) for k in np.load(str(keys_path), allow_pickle=True)]
        X = scipy.sparse.load_npz(str(npz_path))
    if scipy.sparse.issparse(X):
        X = X.toarray()
    return keys, X.astype(np.float32)


def _load_audit_indices(audit_path: Path, top_k: int, tok_sel: str) -> np.ndarray:
    scores: dict[int, float] = {}
    for line in open(audit_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("token_selection") != tok_sel:
            continue
        fid = int(r["feature_idx"])
        sc = float(r.get("score", 0.0))
        if fid not in scores or scores[fid] < sc:
            scores[fid] = sc
    sorted_feats = sorted(scores, key=lambda x: -scores[x])[:top_k]
    return np.array(sorted(sorted_feats), dtype=np.int64)


def _load_gold_labels(predictions_path: Path) -> dict[str, str]:
    """Return {sample_key: gold_label} from probe_val_predictions.jsonl."""
    labels = {}
    for line in open(predictions_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        key = r.get("sample_key", "")
        if key and r.get("gold_label"):
            labels[key] = r["gold_label"]
    return labels


# ── Covariance analysis ───────────────────────────────────────────────────────

def _covariance_analysis(
    I: np.ndarray,   # [N, K] image features
    P: np.ndarray,   # [N, K] prompt features
    labels: np.ndarray,  # [N] bool: True = hateful
) -> dict:
    """Compute Δμ and ΔΣ_IP statistics."""
    n_h = labels.sum()
    n_b = (~labels).sum()
    assert n_h > 0 and n_b > 0, "Need both hateful and benign samples"

    I_h = I[labels]; I_b = I[~labels]
    P_h = P[labels]; P_b = P[~labels]

    mu_I_h = I_h.mean(0); mu_I_b = I_b.mean(0)
    mu_P_h = P_h.mean(0); mu_P_b = P_b.mean(0)
    delta_mu_I = mu_I_h - mu_I_b
    delta_mu_P = mu_P_h - mu_P_b

    # Cross-covariance: E[I P^T] restricted to top-K features → K×K matrix
    # Use efficient batch matmul
    cross_h = (I_h.T @ P_h) / n_h   # [K, K]
    cross_b = (I_b.T @ P_b) / n_b   # [K, K]
    delta_cross = cross_h - cross_b  # ΔΣ_IP (cross-covariance difference)

    # SVD of ΔΣ_IP
    if delta_cross.size == 0 or delta_cross.shape[0] == 0:
        raise ValueError(
            "ΔΣ_IP is empty — no confounder features were selected. "
            "Check --audit_path and --audit_token_selection."
        )
    U, S, Vt = np.linalg.svd(delta_cross, full_matrices=False)

    # Mean shift magnitude vs interaction magnitude
    mean_shift_sq = float(np.dot(delta_mu_I, delta_mu_I) + np.dot(delta_mu_P, delta_mu_P))
    top_singular_sq = float(S[0] ** 2)

    return {
        "n_hateful": int(n_h),
        "n_benign": int(n_b),
        "delta_mu_I": delta_mu_I,
        "delta_mu_P": delta_mu_P,
        "delta_cross": delta_cross,
        "singular_values": S,
        "U": U,   # [K, min(K,K)] — image interaction directions
        "Vt": Vt,  # [min(K,K), K] — prompt interaction directions
        "mean_shift_l2_sq": mean_shift_sq,
        "top_singular_sq": top_singular_sq,
        "interaction_dominance_ratio": top_singular_sq / (mean_shift_sq + 1e-9),
    }


# ── Fire quadrant analysis ────────────────────────────────────────────────────

def _fire_quadrant_stats(
    I: np.ndarray,     # [N, K]
    P: np.ndarray,     # [N, K]
    u: np.ndarray,     # [K] image direction
    v: np.ndarray,     # [K] prompt direction
    labels: np.ndarray,  # [N] bool: True = hateful
    threshold_pct: float = 75.0,
    sample_keys: list[str] | None = None,
    top_k_examples: int = 5,
) -> dict:
    """Compute hateful rate per fire quadrant and collect example keys."""
    s_img = I @ u   # [N]
    s_pmt = P @ v   # [N]

    # Threshold at percentile across all samples
    t_img = float(np.percentile(s_img, threshold_pct))
    t_pmt = float(np.percentile(s_pmt, threshold_pct))

    both  = (s_img > t_img) & (s_pmt > t_pmt)
    img_o = (s_img > t_img) & (s_pmt <= t_pmt)
    pmt_o = (s_img <= t_img) & (s_pmt > t_pmt)
    neith = (s_img <= t_img) & (s_pmt <= t_pmt)

    def stats(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "hateful_rate": None, "examples": []}
        hr = float(labels[mask].mean())
        # Collect top examples by combined score, with gold label
        if sample_keys:
            scores = (s_img + s_pmt)[mask]
            idx_in_mask = np.argsort(-scores)[:top_k_examples]
            orig_idx = np.where(mask)[0][idx_in_mask]
            examples = [
                {"key": sample_keys[i], "gold": "hateful" if labels[i] else "benign",
                 "s_img": round(float(s_img[i]), 4), "s_pmt": round(float(s_pmt[i]), 4)}
                for i in orig_idx
            ]
        else:
            examples = []
        return {"n": n, "hateful_rate": round(hr, 4), "examples": examples}

    return {
        "threshold_img": round(t_img, 5),
        "threshold_pmt": round(t_pmt, 5),
        "both_fire": stats(both),
        "img_only": stats(img_o),
        "pmt_only": stats(pmt_o),
        "neither": stats(neith),
        # Key prediction: hateful_rate(both) >> hateful_rate(one)
        "both_vs_one_delta": round(
            float(labels[both].mean() - labels[img_o | pmt_o].mean())
            if (both.sum() > 0 and (img_o | pmt_o).sum() > 0) else 0.0, 4
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(cfg: FHMInteractionCovarianceConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load feature matrices ──────────────────────────────────────────────────
    logger.info("Loading image-token feature matrix …")
    img_keys, I_full = _load_matrix(
        Path(cfg.image_matrix_path),
        Path(cfg.image_matrix_keys_path) if cfg.image_matrix_keys_path else None,
    )
    logger.info("Loading prompt-token feature matrix …")
    pmt_keys, P_full = _load_matrix(
        Path(cfg.prompt_matrix_path),
        Path(cfg.prompt_matrix_keys_path) if cfg.prompt_matrix_keys_path else None,
    )
    assert img_keys == pmt_keys, (
        f"Key mismatch: {len(img_keys)} image keys vs {len(pmt_keys)} prompt keys. "
        "Ensure both matrices were built from the same dataset run."
    )
    keys = img_keys
    logger.info("Feature matrices: n=%d n_features=%d", len(keys), I_full.shape[1])

    # ── Restrict to top-K confounder features ──────────────────────────────────
    if cfg.audit_path and Path(cfg.audit_path).exists():
        top_feats = _load_audit_indices(
            Path(cfg.audit_path), cfg.bilinear_top_k, cfg.audit_token_selection
        )
        logger.info("Restricting to %d confounder features", len(top_feats))
        I = I_full[:, top_feats]
        P = P_full[:, top_feats]
        feat_indices = [int(x) for x in top_feats]
    else:
        I = I_full; P = P_full
        feat_indices = list(range(I.shape[1]))
        logger.warning("No audit_path — using all %d features", len(feat_indices))

    # ── Load gold labels ──────────────────────────────────────────────────────
    gold_by_key: dict[str, str] = {}
    if cfg.probe_predictions_path and Path(cfg.probe_predictions_path).exists():
        gold_by_key = _load_gold_labels(Path(cfg.probe_predictions_path))
        logger.info("Loaded %d gold labels from probe predictions", len(gold_by_key))
    elif cfg.cache_dir and Path(cfg.cache_dir).exists():
        # Read directly from cache manifest
        from gemma3_experiments.launch.residual.probe_validate import (
            _read_jsonl, _row_gold_label, _row_sample_key,
        )
        manifest = Path(cfg.cache_dir) / "manifest.jsonl"
        assert manifest.exists(), f"Manifest not found: {manifest}"
        for row in _read_jsonl(manifest):
            if row.get("dataset_name") != "Hateful_Memes_Dataset":
                continue
            lbl = _row_gold_label(row)
            key = _row_sample_key(row)
            if lbl and key:
                gold_by_key[key] = lbl
        logger.info("Loaded %d FHM gold labels from cache manifest", len(gold_by_key))
    else:
        raise ValueError("Supply either --probe_predictions_path or --cache_dir for gold labels")

    labels_list = []
    valid_mask = []
    for k in keys:
        lbl = gold_by_key.get(k)
        if lbl is None:
            valid_mask.append(False)
            labels_list.append(False)
        else:
            valid_mask.append(True)
            labels_list.append(lbl == "hateful")

    valid = np.array(valid_mask)
    keys_v = [k for k, v in zip(keys, valid) if v]
    I = I[valid]; P = P[valid]
    labels = np.array(labels_list)[valid]
    logger.info("Labelled samples: %d (hateful=%d benign=%d)", len(keys_v), labels.sum(), (~labels).sum())

    # ── Load feature descriptions ──────────────────────────────────────────────
    descriptions: dict[int, str] = {}
    if cfg.feature_descriptions_path and Path(cfg.feature_descriptions_path).exists():
        raw = json.loads(Path(cfg.feature_descriptions_path).read_text())
        descriptions = {int(k): v for k, v in raw.items() if not isinstance(v, dict)}

    def feat_label(idx: int) -> str:
        fid = feat_indices[idx]
        desc = descriptions.get(fid, "")[:60]
        return f"f{fid}  {desc!r}"

    # ── Covariance analysis ────────────────────────────────────────────────────
    logger.info("Computing covariance-difference analysis …")
    stats = _covariance_analysis(I, P, labels)

    mean_shift_l2 = float(np.sqrt(stats["mean_shift_l2_sq"]))
    S = stats["singular_values"]
    ratio = stats["interaction_dominance_ratio"]

    logger.info("Δμ L2=%.4f  σ_1(ΔΣ_IP)=%.4f  ratio=%.2fx",
                mean_shift_l2, S[0], ratio)

    # ── Interaction feature cards ──────────────────────────────────────────────
    cards = []
    U = stats["U"]
    Vt = stats["Vt"]

    for d in range(min(cfg.top_directions, len(S))):
        u_d = U[:, d]      # image interaction direction
        v_d = Vt[d, :]     # prompt interaction direction

        # Top features by absolute weight
        u_top = np.argsort(-np.abs(u_d))[:cfg.top_k_features]
        v_top = np.argsort(-np.abs(v_d))[:cfg.top_k_features]

        quadrants = _fire_quadrant_stats(
            I, P, u_d, v_d, labels,
            threshold_pct=cfg.threshold_pct,
            sample_keys=keys_v,
            top_k_examples=cfg.top_k_examples,
        )

        cards.append({
            "direction": d,
            "singular_value": round(float(S[d]), 5),
            "image_features": [
                {"idx": int(feat_indices[i]), "weight": round(float(u_d[i]), 5),
                 "description": descriptions.get(feat_indices[i], "")}
                for i in u_top
            ],
            "prompt_features": [
                {"idx": int(feat_indices[i]), "weight": round(float(v_d[i]), 5),
                 "description": descriptions.get(feat_indices[i], "")}
                for i in v_top
            ],
            "quadrants": quadrants,
        })

    # ── Write outputs ──────────────────────────────────────────────────────────
    analysis = {
        "n_samples": int(valid.sum()),
        "n_hateful": int(labels.sum()),
        "n_benign": int((~labels).sum()),
        "n_features_K": len(feat_indices),
        "mean_shift_l2": round(mean_shift_l2, 5),
        "delta_mu_I_l2": round(float(np.linalg.norm(stats["delta_mu_I"])), 5),
        "delta_mu_P_l2": round(float(np.linalg.norm(stats["delta_mu_P"])), 5),
        "singular_values": [round(float(s), 5) for s in S[:cfg.top_directions]],
        "top_singular_sq": round(stats["top_singular_sq"], 5),
        "mean_shift_l2_sq": round(stats["mean_shift_l2_sq"], 5),
        "interaction_dominance_ratio": round(ratio, 3),
        "interaction_cards": cards,
    }
    (output_dir / "fhm_covariance_analysis.json").write_text(json.dumps(analysis, indent=2))
    (output_dir / "fhm_interaction_cards.json").write_text(json.dumps(cards, indent=2))

    # Human-readable report
    lines = [
        "=" * 68,
        "FHM Covariance-Difference Analysis",
        "=" * 68, "",
        f"Samples: {analysis['n_samples']}  hateful={analysis['n_hateful']}  benign={analysis['n_benign']}",
        f"Features: K={analysis['n_features_K']} (top confounder features)",
        "",
        "── Mean-shift vs Interaction strength ──────────────────────────",
        f"  ‖Δμ_I‖ = {analysis['delta_mu_I_l2']:.4f}  (image mean-shift L2)",
        f"  ‖Δμ_P‖ = {analysis['delta_mu_P_l2']:.4f}  (prompt mean-shift L2)",
        f"  ‖Δμ‖²  = {analysis['mean_shift_l2_sq']:.4f}  (total mean-shift energy)",
        f"  σ_1    = {S[0]:.4f}  (top singular value of ΔΣ_IP)",
        f"  σ_1²   = {analysis['top_singular_sq']:.4f}  (top interaction energy)",
        f"  σ_1² / ‖Δμ‖² = {ratio:.2f}×  "
        + ("→ INTERACTION-DOMINATED" if ratio > 1 else "→ MEAN-SHIFT-DOMINATED"),
        "",
        f"  Top {cfg.top_directions} singular values: {[round(float(s),4) for s in S[:cfg.top_directions]]}",
        "",
    ]

    for card in cards:
        d = card["direction"]
        sv = card["singular_value"]
        q = card["quadrants"]
        bfr = q["both_fire"]["hateful_rate"]
        ior = q["img_only"]["hateful_rate"]
        por = q["pmt_only"]["hateful_rate"]
        nr  = q["neither"]["hateful_rate"]
        delta = q["both_vs_one_delta"]

        lines += [
            f"── Direction {d+1}/{cfg.top_directions}  σ={sv:.4f} ─────────────────────────────────",
            "  IMAGE features:",
            *[f"    {feat_label(feat_indices.index(f['idx']) if f['idx'] in feat_indices else 0)}  w={f['weight']:+.4f}"
              for f in card["image_features"][:5]],
            "  PROMPT features:",
            *[f"    {feat_label(feat_indices.index(f['idx']) if f['idx'] in feat_indices else 0)}  w={f['weight']:+.4f}"
              for f in card["prompt_features"][:5]],
            "",
            "  Quadrant analysis (P(hateful) per quadrant):",
            f"    both_fire : n={q['both_fire']['n']:4d}  hateful_rate={bfr}",
            f"    img_only  : n={q['img_only']['n']:4d}  hateful_rate={ior}",
            f"    pmt_only  : n={q['pmt_only']['n']:4d}  hateful_rate={por}",
            f"    neither   : n={q['neither']['n']:4d}  hateful_rate={nr}",
            f"    both_vs_one Δ = {delta:+.4f}  "
            + ("← SUPER-ADDITIVE (interaction causal)" if delta > 0.05 else
               "← additive/subadditive"),
            "",
        ]

    (output_dir / "fhm_covariance_report.txt").write_text("\n".join(lines))
    logger.info("Wrote outputs to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FHMInteractionCovarianceConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
