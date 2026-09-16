

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
import torch
from safetensors import safe_open
from sklearn.preprocessing import MaxAbsScaler
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.probe_fhm_crosscoder import (
    DenseLayerSpec,
    GemmaCrosscoder,
    _manifest_key,
)
from gemma3_experiments.launch.residual.probe_fhm_crosscoder_pairwise import (
    CrosscoderPairwiseReadout,
    FHM_LABELS,
    _L2I,
    _extract_features_cc,
)
from gemma3_experiments.launch.residual.build_fhm_interaction_cards import (
    _build_fhm_pseudo_pairs,
    _manifest_rows,
    _d_model_from_shard,
    _dense_layer_specs,
)
from gemma3_experiments.launch.residual.probe_validate import (
    _calibrate_threshold,
    _per_class_metrics,
    _read_jsonl,
    _row_gold_label,
)
from gemma3_experiments.residual_sae import resolve_visible_device

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BilinearAblationConfig:
    checkpoint_dir: str = ""

    dense_cache_dir: str = ""
    crosscoder_repo_id: str = "google/gemma-scope-2-12b-it"
    crosscoder_hookpoint: str = "resid_post/layer_31_width_65k_l0_medium"
    layer_name: str = "layer31"

    fhm_confounder_metadata_path: str = ""
    output_dir: str = ""

    # Feature extraction
    batch_size: int = 64
    chunk_size: int = 64
    shard_cache_size: int = 16
    aggregation: str = "max"

    # Polarity test: number of permutations to average over
    n_polarity_permutations: int = 100

    device: str = "cuda:0"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_modality(
    rows: list[dict[str, Any]],
    layer_specs: list[DenseLayerSpec],
    crosscoder: GemmaCrosscoder,
    token_selection: str,
    cfg: BilinearAblationConfig,
    device: str,
) -> tuple[list[str], scipy.sparse.csr_matrix]:
    rows_by_layer = [rows]
    key_lists = [[_manifest_key(r) for r in rows]]
    return _extract_features_cc(
        rows_by_layer, key_lists, layer_specs, crosscoder,
        token_selection, cfg.aggregation,
        cfg.batch_size, cfg.chunk_size, cfg.shard_cache_size, device,
    )


def _pair_metrics(
    logits: np.ndarray,
    threshold: float,
    y: list[str],
    keys: list[str],
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    key_to_idx = {k: i for i, k in enumerate(keys)}
    preds = ["hateful" if s >= threshold else "not hateful" for s in logits]
    proba = [{"hateful": float(torch.sigmoid(torch.tensor(s))), "not hateful": 1 - float(torch.sigmoid(torch.tensor(s)))}
             for s in logits]
    pred_by_key = dict(zip(keys, preds))
    proba_by_key = dict(zip(keys, proba))
    pm_rows = []
    for hk, bk in pairs:
        if hk not in key_to_idx or bk not in key_to_idx:
            continue
        hp = proba_by_key[hk]["hateful"]
        bp = proba_by_key[bk]["hateful"]
        pm_rows.append({
            "both_correct": pred_by_key[hk] == "hateful" and pred_by_key[bk] == "not hateful",
            "positive_margin": hp > bp,
            "prob_gap": hp - bp,
        })
    m = _per_class_metrics(y, preds)
    pm = {}
    if pm_rows:
        pm = {
            "n_pairs": len(pm_rows),
            "both_correct_rate": float(np.mean([r["both_correct"] for r in pm_rows])),
            "positive_margin_rate": float(np.mean([r["positive_margin"] for r in pm_rows])),
            "median_prob_gap": float(np.median([r["prob_gap"] for r in pm_rows])),
        }
    return {"global_f1": m.get("macro_f1", 0.0), "global_accuracy": m.get("accuracy", 0.0),
            "pair_metrics": pm}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: BilinearAblationConfig) -> None:
    device = resolve_visible_device(cfg.device)
    ckpt_dir = Path(cfg.checkpoint_dir)
    cache_dir = Path(cfg.dense_cache_dir)
    out_dir   = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    state = torch.load(ckpt_dir / "model_checkpoint.pt", map_location="cpu", weights_only=True)
    feat_indices = json.loads((ckpt_dir / "bilinear_feature_indices.json").read_text())["feature_indices"]
    feat_idx_arr = np.array(feat_indices, dtype=np.int64)
    K = len(feat_indices)

    U = state["bil_U"].numpy()  # [K, r]
    V = state["bil_V"].numpy()  # [K, r]
    r = U.shape[1]
    bil_scale = float(state["bil_scale"].numpy())

    ckpt_metrics = json.loads((ckpt_dir / "pairwise_probe_metrics.json").read_text())
    threshold = float((ckpt_metrics.get("calibrated_metrics") or {}).get("threshold", 0.5))
    saved_cfg = ckpt_metrics.get("config", {})
    img_tok_sel = saved_cfg.get("bilinear_image_token_selection", "image")
    pmt_tok_sel = saved_cfg.get("bilinear_prompt_token_selection", "prompt")
    n_features_full = ckpt_metrics.get("n_features", 65536)
    logger.info("Checkpoint: r=%d K=%d scale=%.4f threshold=%.3f", r, K, bil_scale, threshold)

    # ── Load crosscoder SAE ────────────────────────────────────────────────────
    d_model = _d_model_from_shard(cache_dir)
    layer_specs = _dense_layer_specs(cache_dir, cfg.layer_name)
    logger.info("Loading crosscoder …")
    crosscoder = GemmaCrosscoder.load_from_hub(
        cfg.crosscoder_repo_id, cfg.crosscoder_hookpoint,
        n_layers=len(layer_specs), d_model=d_model, device=device,
    )

    # ── Load manifest rows ─────────────────────────────────────────────────────
    val_rows   = _manifest_rows(cache_dir, ["validation", "val", "dev"], "Hateful_Memes_Dataset")
    train_rows = _manifest_rows(cache_dir, ["train"], "Hateful_Memes_Dataset")
    logger.info("Val: %d rows,  Train: %d rows", len(val_rows), len(train_rows))

    # ── Extract features ──────────────────────────────────────────────────────
    logger.info("Extracting val image/prompt/all features …")
    val_keys_img, X_val_img_raw = _extract_modality(val_rows, layer_specs, crosscoder, img_tok_sel, cfg, device)
    val_keys_pmt, X_val_pmt_raw = _extract_modality(val_rows, layer_specs, crosscoder, pmt_tok_sel, cfg, device)
    val_keys_all, X_val_all_raw = _extract_modality(val_rows, layer_specs, crosscoder, "all", cfg, device)
    assert val_keys_img == val_keys_pmt == val_keys_all

    logger.info("Extracting train image/prompt features for pair analysis …")
    tr_keys_img, X_tr_img_raw = _extract_modality(train_rows, layer_specs, crosscoder, img_tok_sel, cfg, device)
    tr_keys_pmt, X_tr_pmt_raw = _extract_modality(train_rows, layer_specs, crosscoder, pmt_tok_sel, cfg, device)
    assert tr_keys_img == tr_keys_pmt

    # ── Scale features ─────────────────────────────────────────────────────────
    n_tr = len(train_rows)
    all_img_raw = scipy.sparse.vstack([X_tr_img_raw, X_val_img_raw])
    all_pmt_raw = scipy.sparse.vstack([X_tr_pmt_raw, X_val_pmt_raw])
    all_all_raw = scipy.sparse.vstack([
        # need all-token for train too, but only val is needed for linear head
        X_val_all_raw  # just for fitting scaler below
    ])

    scaler_img = MaxAbsScaler()
    scaler_pmt = MaxAbsScaler()
    scaler_all = MaxAbsScaler()
    scaler_img.fit(X_tr_img_raw)
    scaler_pmt.fit(X_tr_pmt_raw)
    scaler_all.fit(X_val_all_raw)  # fit on val only since train all-token not extracted

    X_val_img_top = scaler_img.transform(X_val_img_raw)[:, feat_idx_arr].toarray().astype(np.float32)
    X_val_pmt_top = scaler_pmt.transform(X_val_pmt_raw)[:, feat_idx_arr].toarray().astype(np.float32)
    X_val_all_sc  = scaler_all.transform(X_val_all_raw).toarray().astype(np.float32)
    X_tr_img_top  = scaler_img.transform(X_tr_img_raw)[:, feat_idx_arr].toarray().astype(np.float32)
    X_tr_pmt_top  = scaler_pmt.transform(X_tr_pmt_raw)[:, feat_idx_arr].toarray().astype(np.float32)

    # ── Load full model for val linear-head scores ─────────────────────────────
    logger.info("Running full model on val for linear-head decomposition …")
    model = CrosscoderPairwiseReadout(
        n_features=n_features_full,
        hidden_size=int(saved_cfg.get("hidden_size", 128)),
        gate_hidden=int(saved_cfg.get("gate_hidden_size", 32)),
        dropout=0.0,
        bilinear_k=K, bilinear_rank=r,
    )
    model.load_state_dict(state)
    model.eval().to(device)

    X_val_t   = torch.from_numpy(X_val_all_sc)
    X_val_img_t = torch.from_numpy(X_val_img_top)
    X_val_pmt_t = torch.from_numpy(X_val_pmt_top)

    with torch.no_grad():
        full_logits_val = model(
            X_val_t.to(device), X_val_img_t.to(device), X_val_pmt_t.to(device)
        ).cpu().numpy()
        # Linear-only score (gate*img_head + (1-gate)*text_head, no bilinear)
        g = model.gate(X_val_t.to(device)).squeeze(-1)
        s_img  = model.img_head(X_val_t.to(device)).squeeze(-1)
        s_text = model.text_head(X_val_t.to(device)).squeeze(-1)
        linear_logits_val = (g * s_img + (1 - g) * s_text).cpu().numpy()

    # ── Direction score matrices ────────────────────────────────────────────────
    # val: img_scores_val[n,d] = (X_val_img_top @ U[:,d])[n]
    img_scores_val = X_val_img_top @ U   # [N_val, r]
    pmt_scores_val = X_val_pmt_top @ V   # [N_val, r]
    dir_scores_val = img_scores_val * pmt_scores_val  # [N_val, r]
    bilinear_val   = bil_scale * dir_scores_val.sum(axis=1)  # [N_val]

    # train
    img_scores_tr  = X_tr_img_top @ U   # [N_tr, r]
    pmt_scores_tr  = X_tr_pmt_top @ V   # [N_tr, r]
    dir_scores_tr  = img_scores_tr * pmt_scores_tr  # [N_tr, r]

    val_keys = val_keys_img
    tr_keys  = tr_keys_img
    val_key_to_idx = {k: i for i, k in enumerate(val_keys)}
    tr_key_to_idx  = {k: i for i, k in enumerate(tr_keys)}

    y_val = [next((r["gold_fields"]["label"] for r in val_rows
                   if _manifest_key(r) == k), "not hateful")
             for k in val_keys]
    y_tr = [next((r["gold_fields"]["label"] for r in train_rows
                  if _manifest_key(r) == k), "not hateful")
            for k in tr_keys]

    # ── Pseudo-pairs (train) ───────────────────────────────────────────────────
    pseudo_pairs = _build_fhm_pseudo_pairs(cfg.fhm_confounder_metadata_path, set(tr_keys))
    all_train_pairs: list[tuple[str, str]] = []
    for src_pairs in pseudo_pairs.values():
        all_train_pairs.extend(src_pairs)
    logger.info("Train pseudo-pairs: %s",
                {src: len(p) for src, p in pseudo_pairs.items()})

    # ── Baseline metrics ────────────────────────────────────────────────────────
    baseline_val = _pair_metrics(full_logits_val, threshold, y_val, val_keys, all_train_pairs)
    logger.info("Baseline val: F1=%.4f  (train pairs n=%d)",
                baseline_val["global_f1"], baseline_val["pair_metrics"].get("n_pairs", 0))

    results: dict[str, Any] = {
        "config": {
            "checkpoint_dir": str(ckpt_dir),
            "r": r, "K": K, "bil_scale": bil_scale, "threshold": threshold,
        },
        "baseline": baseline_val,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # INTERVENTION 1: Remove entire bilinear term
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Intervention 1: removing entire bilinear term …")
    no_bil_logits = linear_logits_val  # gate*img_head + (1-gate)*text_head
    no_bil_metrics = _pair_metrics(no_bil_logits, threshold, y_val, val_keys, all_train_pairs)
    # Also calibrate threshold on the no-bilinear scores
    proba_no_bil = np.stack([1 - torch.sigmoid(torch.from_numpy(no_bil_logits)).numpy(),
                             torch.sigmoid(torch.from_numpy(no_bil_logits)).numpy()], axis=1)
    cal_t_no_bil, _, _ = _calibrate_threshold(y_val, proba_no_bil, ["not hateful", "hateful"], "hateful")
    no_bil_cal = _pair_metrics(no_bil_logits, cal_t_no_bil, y_val, val_keys, all_train_pairs)

    results["intervention_1_remove_bilinear"] = {
        "description": "Zero the entire bilinear term; only gate*img_head + (1-gate)*text_head remains",
        "metrics_fixed_threshold": no_bil_metrics,
        "metrics_recalibrated": no_bil_cal,
        "calibrated_threshold": cal_t_no_bil,
        "delta_f1_vs_full": round(no_bil_metrics["global_f1"] - baseline_val["global_f1"], 4),
        "expected_collapse_toward_linear_baseline": "~0.737 (token-pair linear probe)",
    }
    logger.info("  F1 without bilinear: %.4f  (delta=%.4f)",
                no_bil_metrics["global_f1"],
                no_bil_metrics["global_f1"] - baseline_val["global_f1"])

    # ══════════════════════════════════════════════════════════════════════════
    # INTERVENTION 2: Leave-one-rank-out (LORO)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Intervention 2: leave-one-rank-out across %d directions …", r)
    loro_results = []
    for d in range(r):
        # Analytically remove direction d's bilinear contribution from val logits
        dir_d_contribution_val = bil_scale * dir_scores_val[:, d]  # [N_val]
        logits_no_d = full_logits_val - dir_d_contribution_val
        val_m = _pair_metrics(logits_no_d, threshold, y_val, val_keys, all_train_pairs)

        # Train pair metrics: direction d contribution to pair gaps
        dir_d_tr = dir_scores_val[:, d]  # not applicable to train pairs directly
        # Use train-set direction scores for pair-level analysis
        hi_idx = [tr_key_to_idx[hk] for hk, bk in all_train_pairs if hk in tr_key_to_idx and bk in tr_key_to_idx]
        bi_idx = [tr_key_to_idx[bk] for hk, bk in all_train_pairs if hk in tr_key_to_idx and bk in tr_key_to_idx]
        valid_pairs = [(hk, bk) for hk, bk in all_train_pairs
                       if hk in tr_key_to_idx and bk in tr_key_to_idx]
        if hi_idx:
            h_arr, b_arr = np.array(hi_idx), np.array(bi_idx)
            # Direction-d score gap (hateful - benign) for each pair
            d_gap_full = dir_scores_tr[h_arr, d] - dir_scores_tr[b_arr, d]
            # Positive margin rate for this direction
            dir_pos_margin = float(np.mean(d_gap_full > 0))
            dir_mean_gap   = float(np.mean(d_gap_full))
            dir_median_gap = float(np.median(d_gap_full))
        else:
            dir_pos_margin = dir_mean_gap = dir_median_gap = 0.0

        loro_results.append({
            "direction": d + 1,
            "val_f1_no_d": round(val_m["global_f1"], 4),
            "val_f1_delta": round(val_m["global_f1"] - baseline_val["global_f1"], 4),
            "val_n_preds_changed": int(np.sum(
                (logits_no_d >= threshold) != (full_logits_val >= threshold))),
            "train_pair_pos_margin_rate": round(dir_pos_margin, 4),
            "train_pair_mean_gap": round(dir_mean_gap, 5),
            "train_pair_median_gap": round(dir_median_gap, 5),
        })
        if d < 5 or d % 8 == 7:
            logger.info("  LORO dir %2d: val_ΔF1=%+.4f  pair_pos_margin=%.4f",
                        d + 1, loro_results[-1]["val_f1_delta"],
                        loro_results[-1]["train_pair_pos_margin_rate"])

    results["intervention_2_loro"] = {
        "description": "Leave-one-rank-out: zero U[:,d] and V[:,d] for each direction d",
        "per_direction": loro_results,
        "n_directions_with_val_f1_drop": sum(1 for x in loro_results if x["val_f1_delta"] < -0.001),
        "top_direction_by_val_impact": min(loro_results, key=lambda x: x["val_f1_delta"])["direction"],
        "top_direction_by_pair_margin": max(loro_results, key=lambda x: x["train_pair_pos_margin_rate"])["direction"],
    }

    # ══════════════════════════════════════════════════════════════════════════
    # INTERVENTION 3: Factor patching on train pseudo-pairs
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Intervention 3: factor patching on train pseudo-pairs …")
    patch_results = []
    for d in range(r):
        hi_list, bi_list = [], []
        for hk, bk in all_train_pairs:
            if hk in tr_key_to_idx and bk in tr_key_to_idx:
                hi_list.append(tr_key_to_idx[hk])
                bi_list.append(tr_key_to_idx[bk])
        if not hi_list:
            patch_results.append({"direction": d + 1})
            continue
        h_arr = np.array(hi_list)
        b_arr = np.array(bi_list)

        img_h_d = img_scores_tr[h_arr, d]   # hateful image projection, dir d
        img_b_d = img_scores_tr[b_arr, d]   # benign image projection, dir d
        pmt_h_d = pmt_scores_tr[h_arr, d]   # hateful prompt projection, dir d
        pmt_b_d = pmt_scores_tr[b_arr, d]   # benign prompt projection, dir d

        # Bilinear contribution of direction d to benign member
        bil_b_d = bil_scale * img_b_d * pmt_b_d

        # Δ from each patching mode:
        delta_img_patch  = bil_scale * (img_h_d - img_b_d) * pmt_b_d
        delta_pmt_patch  = bil_scale * img_b_d * (pmt_h_d - pmt_b_d)
        delta_both_patch = bil_scale * (img_h_d * pmt_h_d - img_b_d * pmt_b_d)
        cross_term       = delta_both_patch - delta_img_patch - delta_pmt_patch
        # = bil_scale * (img_h_d - img_b_d) * (pmt_h_d - pmt_b_d)

        patch_results.append({
            "direction": d + 1,
            "n_pairs": len(hi_list),
            # Mean delta added to benign score by each patching mode
            "mean_delta_img_patch":  round(float(np.mean(delta_img_patch)), 6),
            "mean_delta_pmt_patch":  round(float(np.mean(delta_pmt_patch)), 6),
            "mean_delta_both_patch": round(float(np.mean(delta_both_patch)), 6),
            "mean_cross_term":       round(float(np.mean(cross_term)), 6),
            # Fraction of pairs where both-patch > max(img-patch, pmt-patch)
            # (super-additivity: bilinear amplifies beyond the stronger single-side patch)
            "superadditivity_rate":  round(float(np.mean(
                delta_both_patch > np.maximum(delta_img_patch, delta_pmt_patch)
            )), 4),
            # Fraction of pairs where cross_term > 0 (bilinear amplification)
            "cross_term_positive_rate": round(float(np.mean(cross_term > 0)), 4),
        })

    # Aggregate: is the cross-term systematically positive?
    mean_cross_all_dirs = float(np.mean([x["mean_cross_term"] for x in patch_results if "mean_cross_term" in x]))
    cross_pos_rate_all  = float(np.mean([x["cross_term_positive_rate"] for x in patch_results if "cross_term_positive_rate" in x]))

    results["intervention_3_factor_patching"] = {
        "description": (
            "For each direction d and each pseudo-pair (hateful h, benign b), patch "
            "img-side, pmt-side, or both from h into b and measure the bilinear score delta. "
            "cross_term = delta_both - delta_img - delta_pmt tests super-additivity."
        ),
        "per_direction": patch_results,
        "aggregate_mean_cross_term": round(mean_cross_all_dirs, 6),
        "aggregate_cross_positive_rate": round(cross_pos_rate_all, 4),
        "interpretation": (
            "If cross_positive_rate >> 0.5, the bilinear mechanism is genuinely multiplicative: "
            "patching both sides amplifies the score beyond the sum of single-side patches. "
            "If cross_positive_rate ≈ 0.5, the mechanism reduces to independent additive effects."
        ),
    }
    logger.info("  Factor patching: mean_cross_term=%.6f  cross_pos_rate=%.4f",
                mean_cross_all_dirs, cross_pos_rate_all)

    # ══════════════════════════════════════════════════════════════════════════
    # INTERVENTION 4: Polarity test (permute image within hateful on train)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Intervention 4: polarity test with %d permutations …",
                cfg.n_polarity_permutations)
    hateful_mask = np.array([1 if _L2I.get(
        next((r["gold_fields"]["label"] for r in train_rows if _manifest_key(r) == k), ""), 0
    ) == 1 else 0 for k in tr_keys], dtype=bool)
    hateful_idx = np.where(hateful_mask)[0]

    rng = np.random.default_rng(42)
    polarity_results = []
    for d in range(r):
        true_scores_h = img_scores_tr[hateful_idx, d] * pmt_scores_tr[hateful_idx, d]
        mean_true = float(np.mean(true_scores_h))

        # Average over multiple permutations to reduce variance
        perm_means = []
        for _ in range(cfg.n_polarity_permutations):
            perm_idx = rng.permutation(hateful_idx)
            perm_scores = img_scores_tr[perm_idx, d] * pmt_scores_tr[hateful_idx, d]
            perm_means.append(float(np.mean(perm_scores)))
        mean_permuted = float(np.mean(perm_means))
        ratio = mean_permuted / mean_true if abs(mean_true) > 1e-9 else float("nan")

        polarity_results.append({
            "direction": d + 1,
            "mean_true_bilinear": round(mean_true, 6),
            "mean_permuted_bilinear": round(mean_permuted, 6),
            "ratio_permuted_over_true": round(ratio, 4),
            # Low ratio → interaction requires specific image-text alignment
            # High ratio → broad harm axis (any hateful image works with any hateful text)
        })

    # Aggregate
    ratios = [x["ratio_permuted_over_true"] for x in polarity_results
              if not np.isnan(x["ratio_permuted_over_true"])]
    results["intervention_4_polarity"] = {
        "description": (
            "For direction d, compare true bilinear score (img_scores[h,d]*pmt_scores[h,d]) "
            "against permuted (shuffle img within hateful, keep pmt fixed). "
            "ratio = permuted/true. Low ratio → specific image×text alignment required."
        ),
        "per_direction": polarity_results,
        "mean_ratio_across_directions": round(float(np.mean(ratios)), 4),
        "median_ratio": round(float(np.median(ratios)), 4),
        "fraction_ratio_below_0.5": round(float(np.mean([x < 0.5 for x in ratios])), 4),
        "interpretation": (
            "ratio < 0.5 for a direction → that direction detects specific image×text conjunctions. "
            "ratio ≈ 1.0 → broad harm axis (any hateful img paired with any hateful text works). "
            "The fraction_ratio_below_0.5 measures how many directions are 'true interactions'."
        ),
    }
    logger.info("  Polarity: mean_ratio=%.4f  median=%.4f  frac<0.5=%.4f",
                results["intervention_4_polarity"]["mean_ratio_across_directions"],
                results["intervention_4_polarity"]["median_ratio"],
                results["intervention_4_polarity"]["fraction_ratio_below_0.5"])

    # ── Save results ───────────────────────────────────────────────────────────
    (out_dir / "bilinear_causal_ablation.json").write_text(json.dumps(results, indent=2))
    logger.info("Wrote results to %s/bilinear_causal_ablation.json", out_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("FHM Bilinear Causal Ablation Summary")
    print(f"Checkpoint: {ckpt_dir.name}  r={r}  K={K}  scale={bil_scale:.4f}  thr={threshold:.3f}")
    print(f"\nBaseline val F1: {baseline_val['global_f1']:.4f}")
    print(f"\nIntervention 1 — Remove bilinear entirely:")
    print(f"  Val F1 (fixed thr):   {results['intervention_1_remove_bilinear']['metrics_fixed_threshold']['global_f1']:.4f}  "
          f"(ΔF1={results['intervention_1_remove_bilinear']['delta_f1_vs_full']:+.4f})")
    print(f"  Val F1 (recalibrated): {results['intervention_1_remove_bilinear']['metrics_recalibrated']['global_f1']:.4f}")
    print(f"\nIntervention 2 — LORO: {results['intervention_2_loro']['n_directions_with_val_f1_drop']} directions "
          f"cause F1 drop; top ablation=dir {results['intervention_2_loro']['top_direction_by_val_impact']}, "
          f"top pair-margin=dir {results['intervention_2_loro']['top_direction_by_pair_margin']}")
    print(f"\nIntervention 3 — Factor patching: mean_cross_term={mean_cross_all_dirs:.6f}  "
          f"cross_positive_rate={cross_pos_rate_all:.4f}")
    print(f"\nIntervention 4 — Polarity: mean_ratio={results['intervention_4_polarity']['mean_ratio_across_directions']:.4f}  "
          f"frac_below_0.5={results['intervention_4_polarity']['fraction_ratio_below_0.5']:.4f}")
    print("=" * 70)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(BilinearAblationConfig,
                argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
