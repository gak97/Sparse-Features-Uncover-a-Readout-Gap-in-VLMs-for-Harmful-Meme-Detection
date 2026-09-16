

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from safetensors import safe_open
from sklearn.preprocessing import MaxAbsScaler
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.probe_fhm_crosscoder import (
    DenseLayerSpec,
    DenseShardReader,
    GemmaCrosscoder,
    _aggregate_acts,
    _common_keys,
    _manifest_key,
    _role_mask,
)
from gemma3_experiments.launch.residual.probe_fhm_crosscoder_pairwise import (
    CrosscoderPairwiseReadout,
    FHM_LABELS,
    _L2I,
    _extract_features_cc,
    _load_bilinear_feature_indices,
)
from gemma3_experiments.launch.residual.probe_validate import (
    _calibrate_threshold,
    _load_fhm_confounder_metadata,
    _normalise_fhm_id,
    _per_class_metrics,
    _read_jsonl,
    _row_gold_label,
    _row_sample_key,
)
from gemma3_experiments.residual_sae import resolve_visible_device

logger = logging.getLogger(__name__)

TASK_KEY = "Hateful_Memes_Dataset::hateful_memes_binary::label"


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InteractionCardConfig:
    # Checkpoint from probe_fhm_crosscoder_pairwise (must have save_checkpoint=True)
    checkpoint_dir: str = ""

    # Dense activation cache and crosscoder SAE (same as used for training)
    dense_cache_dir: str = ""
    crosscoder_repo_id: str = "google/gemma-scope-2-12b-it"
    crosscoder_hookpoint: str = "resid_post/layer_31_width_65k_l0_medium"
    layer_name: str = "layer31"

    # IVL pseudo-pair metadata (ivl_plus_gemini_captions_complete.json)
    fhm_confounder_metadata_path: str = ""

    # Feature semantics
    feature_descriptions_path: str = ""
    audit_path: str = ""

    # Output
    output_dir: str = ""

    # Card verbosity
    top_k_features: int = 10     # features shown per U/V column
    top_k_pairs: int = 10        # top product pair combinations shown
    top_k_examples: int = 15     # top activating examples per direction
    n_counterexamples: int = 10  # high-img/low-pmt counterexamples (and vice versa)

    # Number of strongest directions to include in the curated interaction_report.txt
    # (all directions are still written to interaction_cards.json / direction_summary.json).
    top_n_cards: int = 12

    # Feature extraction (same defaults as pairwise training)
    batch_size: int = 64
    chunk_size: int = 64
    shard_cache_size: int = 16
    aggregation: str = "max"

    # Ablation calibration: use the same threshold from the checkpoint's calib split
    calibration_frac: float = 0.20
    calibration_seed: int = 42

    pair_analysis_splits: list[str] = field(default_factory=lambda: ["train"])

    device: str = "cuda:0"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_descriptions(path: str) -> dict[int, str]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {int(k): v for k, v in raw.items() if not isinstance(v, dict)}


def _load_audit_scores(path: str) -> dict[int, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    scores: dict[int, dict[str, Any]] = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        fid = int(r["feature_idx"])
        sc = float(r.get("score", 0.0))
        if fid not in scores or sc > scores[fid].get("score", 0.0):
            scores[fid] = {"score": sc,
                           "direction": r.get("direction", ""),
                           "sign_consistency": float(r.get("sign_consistency", 0.0))}
    return scores


def _manifest_rows(cache_dir: Path, splits: list[str], dataset: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for line in open(cache_dir / "manifest.jsonl"):
        r = json.loads(line)
        if splits and r.get("split") not in splits:
            continue
        if dataset and r.get("dataset_name") != dataset:
            continue
        r["_cache_dir"] = str(cache_dir)
        rows.append(r)
    return rows


def _dense_layer_specs(cache_dir: Path, layer_name: str) -> list[DenseLayerSpec]:
    cache_str = str(cache_dir)
    return [DenseLayerSpec(layer_name, cache_str, cache_str)]


def _extract_modality_features(
    rows: list[dict[str, Any]],
    layer_specs: list[DenseLayerSpec],
    crosscoder: GemmaCrosscoder,
    token_selection: str,
    cfg: InteractionCardConfig,
    device: str,
) -> tuple[list[str], scipy.sparse.csr_matrix]:
    """Return (keys, sparse CSR feature matrix) for the given token selection."""
    rows_by_layer = [rows]
    key_lists = [[_manifest_key(r) for r in rows]]
    return _extract_features_cc(
        rows_by_layer, key_lists, layer_specs, crosscoder,
        token_selection, cfg.aggregation,
        cfg.batch_size, cfg.chunk_size, cfg.shard_cache_size, device,
    )


def _d_model_from_shard(cache_dir: Path) -> int:
    """Read hidden-state dimension from the first shard tensor."""
    shards = sorted(cache_dir.glob("*.safetensors"))
    assert shards, f"No safetensors shards found in {cache_dir}"
    with safe_open(str(shards[0]), framework="pt", device="cpu") as fd:
        return int(fd.get_tensor("hidden").shape[1])


def _build_fhm_pseudo_pairs(
    meta_path: str,
    key_set: set[str],
) -> dict[str, list[tuple[str, str]]]:
    """Return {source: [(hateful_key, benign_key), ...]} filtered to available keys."""
    if not meta_path or not Path(meta_path).exists():
        return {}
    meta = _load_fhm_confounder_metadata(Path(meta_path))
    pairs_by_source: dict[str, list[tuple[str, str]]] = {}
    for src in ("pseudo_img", "pseudo_text"):
        idx_field = "pseudo_img_idx" if src == "pseudo_img" else "pseudo_text_idx"
        groups: dict[str, list[dict[str, Any]]] = {}
        for sample_id, m in meta.items():
            grp = m.get(idx_field)
            # Skip null/sentinel values: IVL uses 0 / 0.0 for "no group"
            if grp is None or grp == 0 or grp == 0.0:
                continue
            grp = str(int(float(grp)))   # normalise float group IDs (e.g. 7160.0 → "7160")
            groups.setdefault(grp, []).append({"sample_id": sample_id, **m})
        pairs = []
        for grp_rows in groups.values():
            # IVL uses integer label (1=hateful, 0=not hateful); manifest gold_fields uses text
            h = [r for r in grp_rows if int(r.get("label", r.get("gold_label") == "hateful")) == 1]
            b = [r for r in grp_rows if int(r.get("label", r.get("gold_label") == "hateful")) == 0]
            for hr in h:
                for br in b:
                    hk = f"Hateful_Memes_Dataset::hateful_memes_binary::{_normalise_fhm_id(hr['sample_id'])}"
                    bk = f"Hateful_Memes_Dataset::hateful_memes_binary::{_normalise_fhm_id(br['sample_id'])}"
                    if hk in key_set and bk in key_set:
                        pairs.append((hk, bk))
        pairs_by_source[src] = pairs
    return pairs_by_source


def _build_global_word_doc_freq(descriptions: dict[int, str]) -> tuple[dict[str, int], int]:
    pass
    import re
    from collections import Counter
    doc_freq: Counter[str] = Counter()
    n_docs = 0
    for desc in descriptions.values():
        if not isinstance(desc, str) or not desc:
            continue
        n_docs += 1
        words = set(re.findall(r"[a-zA-Z]{4,}", desc.lower()))
        doc_freq.update(words)
    return dict(doc_freq), max(n_docs, 1)


def _suggest_label(
    u_top_descs: list[str],
    v_top_descs: list[str],
    global_doc_freq: dict[str, int] | None = None,
    global_n_docs: int = 1,
) -> str:
    pass
    import math
    import re
    from collections import Counter
    # Strip common LLM boilerplate prefixes before tokenising
    boilerplate = re.compile(
        r"^(this feature (detects|activates on|encodes|responds to|fires on|identifies)"
        r"|the feature (detects|activates|encodes|responds)|feature \d+:?)\s*",
        re.IGNORECASE,
    )
    stop = {"of", "the", "a", "an", "in", "or", "and", "to", "for", "with",
            "is", "are", "be", "has", "that", "this", "on", "at", "by", "from",
            "about", "no", "any", "which", "as", "its", "it", "their", "also",
            "not", "but", "can", "may", "more", "other", "some", "they", "these",
            "often", "such", "feature", "detects", "encodes", "content", "memes",
            "text", "pattern", "harmful", "hateful", "activates", "including",
            "even", "then", "were", "true", "was", "had", "did", "him", "her",
            "them", "those", "being", "having", "into", "than", "when", "where",
            "while", "yes", "would", "could", "should", "will", "does", "used",
            "uses", "using", "like", "similar", "related", "associated", "involving"}
    all_words = []
    for d in u_top_descs + v_top_descs:
        if d:
            d = boilerplate.sub("", d.strip())
            all_words.extend(re.findall(r"[a-zA-Z]{4,}", d.lower()))
    local_counts = Counter(w for w in all_words if w not in stop)
    if not local_counts:
        return "(no descriptions available)"
    if global_doc_freq:
        scored = []
        for w, tf in local_counts.items():
            df = global_doc_freq.get(w, 1)
            idf = math.log((global_n_docs + 1) / (df + 1)) + 1.0
            scored.append((tf * idf, w))
        scored.sort(key=lambda x: -x[0])
        top = [w for _, w in scored[:6]]
    else:
        top = [w for w, _ in local_counts.most_common(6)]
    return " / ".join(top[:4])


# ──────────────────────────────────────────────────────────────────────────────
# Per-direction card computation
# ──────────────────────────────────────────────────────────────────────────────

def _build_card(
    d: int,
    U: np.ndarray,          # [K, r]
    V: np.ndarray,          # [K, r]
    feat_indices: list[int],
    img_scores: np.ndarray,  # [N_val] direction-d image score per sample
    pmt_scores: np.ndarray,  # [N_val] direction-d prompt score per sample
    dir_scores: np.ndarray,  # [N_val] = img_scores * pmt_scores
    full_logits: np.ndarray, # [N_val] full model logit (pre-threshold)
    bil_scale: float,
    threshold: float,
    val_keys: list[str],
    y_val: list[str],        # gold labels for val
    meta_by_key: dict[str, dict[str, Any]],  # manifest row by key
    pseudo_pairs: dict[str, list[tuple[str, str]]],
    key_to_val_idx: dict[str, int],  # maps sample key → index into pair_dir_scores
    pair_dir_scores: np.ndarray,     # [N_pair] direction scores for pair-analysis rows
    descriptions: dict[int, str],
    audit_scores: dict[int, dict[str, Any]],
    top_k_features: int,
    top_k_pairs: int,
    top_k_examples: int,
    n_counterexamples: int,
    global_doc_freq: dict[str, int] | None = None,
    global_n_docs: int = 1,
) -> dict[str, Any]:

    u_col = U[:, d]  # [K]
    v_col = V[:, d]  # [K]
    K = len(feat_indices)
    strength = float(np.linalg.norm(u_col) * np.linalg.norm(v_col))

    # ── 1. Top U (image) features ─────────────────────────────────────────────
    u_order = np.argsort(-np.abs(u_col))[:top_k_features]
    u_features = []
    for ki in u_order:
        fid = feat_indices[ki]
        u_features.append({
            "rank": int(ki),
            "feature_idx": int(fid),
            "weight": round(float(u_col[ki]), 5),
            "abs_weight": round(float(abs(u_col[ki])), 5),
            "description": descriptions.get(fid, ""),
            "sign_consistency": round(audit_scores.get(fid, {}).get("sign_consistency", 0.0), 4),
            "audit_score": round(audit_scores.get(fid, {}).get("score", 0.0), 4),
        })

    # ── 2. Top V (prompt) features ────────────────────────────────────────────
    v_order = np.argsort(-np.abs(v_col))[:top_k_features]
    v_features = []
    for ki in v_order:
        fid = feat_indices[ki]
        v_features.append({
            "rank": int(ki),
            "feature_idx": int(fid),
            "weight": round(float(v_col[ki]), 5),
            "abs_weight": round(float(abs(v_col[ki])), 5),
            "description": descriptions.get(fid, ""),
            "sign_consistency": round(audit_scores.get(fid, {}).get("sign_consistency", 0.0), 4),
            "audit_score": round(audit_scores.get(fid, {}).get("score", 0.0), 4),
        })

    # ── 3. Top product pairs (feature_i × feature_j) ─────────────────────────
    # Consider top-20 img and top-20 pmt features to keep pair search tractable
    u_top20 = np.argsort(-np.abs(u_col))[:min(20, K)]
    v_top20 = np.argsort(-np.abs(v_col))[:min(20, K)]
    pair_list: list[tuple[float, int, int]] = []
    for ui in u_top20:
        for vi in v_top20:
            product = float(u_col[ui] * v_col[vi])
            pair_list.append((product, int(ui), int(vi)))
    pair_list.sort(key=lambda x: -abs(x[0]))

    def _pair_entry(product: float, ui: int, vi: int) -> dict[str, Any]:
        fi, fj = feat_indices[ui], feat_indices[vi]
        return {
            "product": round(product, 5),
            "image_feature": int(fi),
            "image_feature_weight": round(float(u_col[ui]), 5),
            "image_description": descriptions.get(fi, ""),
            "prompt_feature": int(fj),
            "prompt_feature_weight": round(float(v_col[vi]), 5),
            "prompt_description": descriptions.get(fj, ""),
            "interpretation": (
                "hateful when image feature fires AND prompt feature fires"
                if product > 0 else
                "hateful when image feature fires BUT NOT prompt feature (contrast)"
            ),
        }

    pos_pairs = [_pair_entry(p, ui, vi) for p, ui, vi in pair_list if p > 0][:top_k_pairs]
    neg_pairs = [_pair_entry(p, ui, vi) for p, ui, vi in pair_list if p < 0][:top_k_pairs]

    # ── 4. Top activating examples ────────────────────────────────────────────
    order_desc = np.argsort(-dir_scores)  # highest direction score first
    order_asc  = np.argsort( dir_scores)  # lowest (negative) direction score first

    def _example(idx: int) -> dict[str, Any]:
        key = val_keys[idx]
        row = meta_by_key.get(key, {})
        pred = "hateful" if full_logits[idx] >= threshold else "not hateful"
        return {
            "key": key,
            "gold_label": y_val[idx],
            "predicted_label": pred,
            "correct": pred == y_val[idx],
            "direction_score": round(float(dir_scores[idx]), 5),
            "img_component": round(float(img_scores[idx]), 5),
            "pmt_component": round(float(pmt_scores[idx]), 5),
            "meme_text": row.get("meme_text", ""),
            "image_path": row.get("image_path", ""),
            "generated_text": (row.get("generated_text", "") or "")[:300],
        }

    n_ex = min(top_k_examples, len(val_keys))
    top_examples = [_example(i) for i in order_desc[:n_ex]]
    bot_examples = [_example(i) for i in order_asc[:min(5, len(val_keys))]]

    # High-score hateful examples and high-score benign examples separately
    hateful_idx = [i for i in order_desc if y_val[i] == "hateful"]
    benign_idx  = [i for i in order_desc if y_val[i] == "not hateful"]
    top_hateful = [_example(i) for i in hateful_idx[:min(10, len(hateful_idx))]]
    top_benign  = [_example(i) for i in benign_idx[:min(5, len(benign_idx))]]

    # ── 5. Pseudo-pair deltas ─────────────────────────────────────────────────
    # key_to_val_idx indexes into pair_dir_scores (may be train rows)
    pair_stats: dict[str, Any] = {}
    for src, pairs in pseudo_pairs.items():
        deltas, pos_margins = [], []
        for hk, bk in pairs:
            hi = key_to_val_idx.get(hk)
            bi = key_to_val_idx.get(bk)
            if hi is None or bi is None:
                continue
            delta = float(pair_dir_scores[hi] - pair_dir_scores[bi])
            deltas.append(delta)
            pos_margins.append(delta > 0)
        if deltas:
            pair_stats[src] = {
                "n_pairs": len(deltas),
                "mean_delta": round(float(np.mean(deltas)), 5),
                "median_delta": round(float(np.median(deltas)), 5),
                "positive_margin_rate": round(float(np.mean(pos_margins)), 4),
            }

    # ── 6. Counterexamples ────────────────────────────────────────────────────
    # "Visual-only fire": high img_score, low pmt_score
    img_q75 = np.percentile(img_scores, 75)
    pmt_q25 = np.percentile(pmt_scores, 25)
    pmt_q75 = np.percentile(pmt_scores, 75)
    img_q25 = np.percentile(img_scores, 25)

    vis_only_mask = (img_scores >= img_q75) & (pmt_scores <= pmt_q25)
    txt_only_mask = (pmt_scores >= pmt_q75) & (img_scores <= img_q25)

    vis_only_idx = np.where(vis_only_mask)[0]
    txt_only_idx = np.where(txt_only_mask)[0]
    np.random.seed(0)
    if len(vis_only_idx) > n_counterexamples:
        vis_only_idx = np.random.choice(vis_only_idx, n_counterexamples, replace=False)
    if len(txt_only_idx) > n_counterexamples:
        txt_only_idx = np.random.choice(txt_only_idx, n_counterexamples, replace=False)

    counterexamples = {
        "visual_only_fires": [_example(i) for i in vis_only_idx],
        "text_only_fires": [_example(i) for i in txt_only_idx],
        "visual_only_label_breakdown": {
            "hateful": int(sum(1 for i in vis_only_idx if y_val[i] == "hateful")),
            "not_hateful": int(sum(1 for i in vis_only_idx if y_val[i] == "not hateful")),
        },
        "text_only_label_breakdown": {
            "hateful": int(sum(1 for i in txt_only_idx if y_val[i] == "hateful")),
            "not_hateful": int(sum(1 for i in txt_only_idx if y_val[i] == "not hateful")),
        },
    }

    # ── 7. Ablation (analytical) ──────────────────────────────────────────────
    # Remove direction d's contribution from the total logit.
    # full_logit[n] = gate*img_head + (1-gate)*text_head + scale * sum_d(img_d * pmt_d)
    # ablated_logit[n] = full_logit[n] - scale * dir_scores[n]
    ablated_logits = full_logits - bil_scale * dir_scores
    full_preds    = ["hateful" if s >= threshold else "not hateful" for s in full_logits]
    ablated_preds = ["hateful" if s >= threshold else "not hateful" for s in ablated_logits]
    full_metrics    = _per_class_metrics(y_val, full_preds)
    ablated_metrics = _per_class_metrics(y_val, ablated_preds)
    n_changed = int(sum(1 for fp, ap in zip(full_preds, ablated_preds) if fp != ap))
    ablation = {
        "full_macro_f1": round(float(full_metrics.get("macro_f1", 0.0)), 4),
        "ablated_macro_f1": round(float(ablated_metrics.get("macro_f1", 0.0)), 4),
        "delta_macro_f1": round(float(ablated_metrics.get("macro_f1", 0.0)) -
                                float(full_metrics.get("macro_f1", 0.0)), 4),
        "n_predictions_changed": n_changed,
        "n_val": len(y_val),
    }

    # ── 8. Candidate semantic label ───────────────────────────────────────────
    u_descs = [f["description"] for f in u_features if f["description"]]
    v_descs = [f["description"] for f in v_features if f["description"]]
    candidate_label = _suggest_label(u_descs, v_descs, global_doc_freq, global_n_docs)

    return {
        "direction": d,
        "direction_1indexed": d + 1,
        "strength": round(strength, 5),
        "u_norm": round(float(np.linalg.norm(u_col)), 5),
        "v_norm": round(float(np.linalg.norm(v_col)), 5),
        "candidate_semantic_label": candidate_label,
        "u_features_image_side": u_features,
        "v_features_prompt_side": v_features,
        "top_positive_pairs": pos_pairs,
        "top_negative_pairs": neg_pairs,
        "top_activating_examples": top_examples,
        "top_hateful_examples": top_hateful,
        "top_benign_examples_by_direction": top_benign,
        "bottom_activating_examples": bot_examples,
        "pseudo_pair_deltas": pair_stats,
        "counterexamples": counterexamples,
        "ablation": ablation,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Text report formatting
# ──────────────────────────────────────────────────────────────────────────────

def _format_card_txt(card: dict[str, Any]) -> str:
    lines = []
    sep = "═" * 70

    lines.append(sep)
    lines.append(f"Direction {card['direction_1indexed']}/{card.get('total_directions', '?')}  "
                 f"(strength={card['strength']:.4f}  |U|={card['u_norm']:.3f}  |V|={card['v_norm']:.3f})")
    lines.append(f"Candidate label: {card['candidate_semantic_label']}")
    lines.append("")

    lines.append("── IMAGE-SIDE features (U[:,d]) ──────────────────────────────────")
    for f in card["u_features_image_side"]:
        desc = (f["description"] or "(no description)")[:70]
        lines.append(f"  f{f['feature_idx']:6d}  w={f['weight']:+.4f}  "
                     f"sc={f['sign_consistency']:.3f}  {desc!r}")

    lines.append("")
    lines.append("── PROMPT-SIDE features (V[:,d]) ─────────────────────────────────")
    for f in card["v_features_prompt_side"]:
        desc = (f["description"] or "(no description)")[:70]
        lines.append(f"  f{f['feature_idx']:6d}  w={f['weight']:+.4f}  "
                     f"sc={f['sign_consistency']:.3f}  {desc!r}")

    lines.append("")
    lines.append("── TOP PRODUCT PAIRS (image_feature × prompt_feature) ────────────")
    lines.append("  POSITIVE (both fire → hateful):")
    for p in card["top_positive_pairs"][:5]:
        di = (p["image_description"] or "(no desc)")[:40]
        dj = (p["prompt_description"] or "(no desc)")[:40]
        lines.append(f"    f{p['image_feature']}×f{p['prompt_feature']}  "
                     f"product={p['product']:+.4f}  [{di!r}] × [{dj!r}]")
    lines.append("  NEGATIVE (image fires but NOT prompt → hateful, contrast):")
    for p in card["top_negative_pairs"][:5]:
        di = (p["image_description"] or "(no desc)")[:40]
        dj = (p["prompt_description"] or "(no desc)")[:40]
        lines.append(f"    f{p['image_feature']}×f{p['prompt_feature']}  "
                     f"product={p['product']:+.4f}  [{di!r}] × [{dj!r}]")

    lines.append("")
    lines.append("── TOP ACTIVATING EXAMPLES ───────────────────────────────────────")
    for ex in card["top_activating_examples"][:8]:
        tick = "✓" if ex["correct"] else "✗"
        lines.append(f"  [{tick}] {ex['gold_label']:12s} d={ex['direction_score']:+.4f} "
                     f"(img={ex['img_component']:+.3f} pmt={ex['pmt_component']:+.3f})")
        lines.append(f"      Text: {ex['meme_text'][:100]!r}")

    lines.append("")
    lines.append("── COUNTEREXAMPLES ───────────────────────────────────────────────")
    lines.append("  Visual-only fires (high img, low pmt):")
    for ex in card["counterexamples"]["visual_only_fires"][:4]:
        tick = "✓" if ex["correct"] else "✗"
        lines.append(f"    [{tick}] {ex['gold_label']:12s} {ex['meme_text'][:80]!r}")
    lines.append("  Text-only fires (high pmt, low img):")
    for ex in card["counterexamples"]["text_only_fires"][:4]:
        tick = "✓" if ex["correct"] else "✗"
        lines.append(f"    [{tick}] {ex['gold_label']:12s} {ex['meme_text'][:80]!r}")

    lines.append("")
    lines.append("── PSEUDO-PAIR DELTAS ────────────────────────────────────────────")
    for src, stats in card["pseudo_pair_deltas"].items():
        lines.append(f"  {src}: n={stats['n_pairs']}  "
                     f"mean_delta={stats['mean_delta']:+.4f}  "
                     f"positive_margin={stats['positive_margin_rate']:.3f}")

    lines.append("")
    lines.append("── ABLATION ──────────────────────────────────────────────────────")
    ab = card["ablation"]
    lines.append(f"  Remove direction: F1 {ab['full_macro_f1']:.4f} → {ab['ablated_macro_f1']:.4f}  "
                 f"(Δ={ab['delta_macro_f1']:+.4f},  {ab['n_predictions_changed']} predictions changed)")

    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(cfg: InteractionCardConfig) -> None:
    device = resolve_visible_device(cfg.device)
    ckpt_dir = Path(cfg.checkpoint_dir)
    cache_dir = Path(cfg.dense_cache_dir)
    out_dir   = Path(cfg.output_dir) if cfg.output_dir else (ckpt_dir / "interaction_cards")
    out_dir.mkdir(parents=True, exist_ok=True)

    assert (ckpt_dir / "model_checkpoint.pt").exists(), "model_checkpoint.pt not found — run probe with save_checkpoint=True"
    assert (ckpt_dir / "bilinear_feature_indices.json").exists(), "bilinear_feature_indices.json not found"

    # ── Load checkpoint ────────────────────────────────────────────────────────
    logger.info("Loading checkpoint from %s", ckpt_dir)
    state = torch.load(ckpt_dir / "model_checkpoint.pt", map_location="cpu", weights_only=True)
    feat_indices = json.loads((ckpt_dir / "bilinear_feature_indices.json").read_text())["feature_indices"]
    K = len(feat_indices)

    assert "bil_U" in state, "Checkpoint has no bil_U — only low-rank bilinear supports interaction cards"
    U = state["bil_U"].numpy()  # [K, r]
    V = state["bil_V"].numpy()  # [K, r]
    r = U.shape[1]
    bil_scale = float(state["bil_scale"].numpy())
    logger.info("Loaded U[%d×%d], V[%d×%d], scale=%.4f", K, r, K, r, bil_scale)

    # Load metrics for threshold and config
    metrics_path = ckpt_dir / "pairwise_probe_metrics.json"
    ckpt_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    threshold = float((ckpt_metrics.get("calibrated_metrics") or {}).get("threshold", 0.5))
    logger.info("Threshold from checkpoint calib split: %.3f", threshold)

    saved_cfg = ckpt_metrics.get("config", {})
    img_tok_sel = saved_cfg.get("bilinear_image_token_selection", "image")
    pmt_tok_sel = saved_cfg.get("bilinear_prompt_token_selection", "prompt")
    audit_tok_sel = saved_cfg.get("bilinear_audit_token_selection", "prompt_and_image")

    # ── Load feature descriptions and audit scores ────────────────────────────
    descriptions = _load_descriptions(cfg.feature_descriptions_path)
    audit_sc     = _load_audit_scores(cfg.audit_path)
    logger.info("Loaded %d descriptions, %d audit scores", len(descriptions), len(audit_sc))
    global_doc_freq, global_n_docs = _build_global_word_doc_freq(descriptions)
    logger.info("Built IDF background corpus from %d descriptions (%d unique words)",
                global_n_docs, len(global_doc_freq))

    # ── Load crosscoder SAE ────────────────────────────────────────────────────
    logger.info("Loading crosscoder SAE: %s  %s", cfg.crosscoder_repo_id, cfg.crosscoder_hookpoint)
    d_model = _d_model_from_shard(cache_dir)
    layer_specs = _dense_layer_specs(cache_dir, cfg.layer_name)
    crosscoder = GemmaCrosscoder.load_from_hub(
        cfg.crosscoder_repo_id, cfg.crosscoder_hookpoint,
        n_layers=len(layer_specs), d_model=d_model, device=device,
    )

    # ── Load FHM manifest rows ────────────────────────────────────────────────
    train_rows = _manifest_rows(cache_dir, ["train"], "Hateful_Memes_Dataset")
    val_rows   = _manifest_rows(cache_dir, ["validation", "val", "dev"], "Hateful_Memes_Dataset")
    all_rows   = train_rows + val_rows
    logger.info("FHM: %d train rows, %d val rows", len(train_rows), len(val_rows))

    # ── Extract image-token and prompt-token feature matrices ─────────────────
    logger.info("Extracting image-token features (token_selection=%s) …", img_tok_sel)
    all_keys_img, X_all_img = _extract_modality_features(
        all_rows, layer_specs, crosscoder, img_tok_sel, cfg, device)

    logger.info("Extracting prompt-token features (token_selection=%s) …", pmt_tok_sel)
    all_keys_pmt, X_all_pmt = _extract_modality_features(
        all_rows, layer_specs, crosscoder, pmt_tok_sel, cfg, device)

    # Align keys (should be identical ordering but be safe)
    assert all_keys_img == all_keys_pmt, "Image and prompt key lists are misaligned"
    all_keys = all_keys_img

    # ── Scale using MaxAbsScaler fitted on train rows ─────────────────────────
    n_train = len(train_rows)
    scaler_img = MaxAbsScaler()
    scaler_pmt = MaxAbsScaler()
    X_tr_img = scaler_img.fit_transform(X_all_img[:n_train])
    X_tr_pmt = scaler_pmt.fit_transform(X_all_pmt[:n_train])
    X_all_img_sc = scaler_img.transform(X_all_img)
    X_all_pmt_sc = scaler_pmt.transform(X_all_pmt)

    # Slice to top-K bilinear feature indices
    feat_idx_arr = np.array(feat_indices, dtype=np.int64)
    X_all_img_top = X_all_img_sc[:, feat_idx_arr].toarray().astype(np.float32)  # [N, K]
    X_all_pmt_top = X_all_pmt_sc[:, feat_idx_arr].toarray().astype(np.float32)  # [N, K]

    # Split train/val
    X_val_img_top = X_all_img_top[n_train:]  # [N_val, K]
    X_val_pmt_top = X_all_pmt_top[n_train:]  # [N_val, K]
    val_keys = all_keys[n_train:]
    key_to_val_idx = {k: i for i, k in enumerate(val_keys)}
    logger.info("Val keys: %d", len(val_keys))

    # ── Load full model for inference (gate + head scores) ───────────────────
    logger.info("Loading full model for inference …")
    metrics_cfg = ckpt_metrics.get("config", {})
    n_features = crosscoder.num_latents

    # We need the all-token features for the model's main (non-bilinear) heads
    logger.info("Extracting all-token features for full model inference …")
    all_keys_all, X_all_all = _extract_modality_features(
        all_rows, layer_specs, crosscoder, "all", cfg, device)
    assert all_keys_all == all_keys
    scaler_all = MaxAbsScaler()
    scaler_all.fit(X_all_all[:n_train])
    X_all_all_sc = scaler_all.transform(X_all_all)
    X_val_all = torch.from_numpy(X_all_all_sc[n_train:].toarray().astype(np.float32))
    X_val_img_t = torch.from_numpy(X_val_img_top)
    X_val_pmt_t = torch.from_numpy(X_val_pmt_top)

    model = CrosscoderPairwiseReadout(
        n_features=n_features,
        hidden_size=int(metrics_cfg.get("hidden_size", 128)),
        gate_hidden=int(metrics_cfg.get("gate_hidden_size", 32)),
        dropout=0.0,  # eval mode
        bilinear_k=K,
        bilinear_rank=r,
        triple_bilinear=bool(metrics_cfg.get("use_triple_bilinear", False)),
    )
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    with torch.no_grad():
        logits = model(X_val_all.to(device), X_val_img_t.to(device), X_val_pmt_t.to(device))
        full_logits_np = logits.cpu().numpy()  # [N_val] raw logits (pre-sigmoid)

    # ── Gold labels and manifest metadata for val ─────────────────────────────
    val_manifest_by_key: dict[str, dict[str, Any]] = {}
    for row in val_rows:
        key = _manifest_key(row)
        val_manifest_by_key[key] = row

    y_val = []
    for key in val_keys:
        row = val_manifest_by_key.get(key, {})
        gf = row.get("gold_fields", {})
        label = gf.get("label", "")
        y_val.append(label if label in FHM_LABELS else "not hateful")

    # ── Per-direction direction scores ────────────────────────────────────────
    # X_val_img_top @ U[:,d] and X_val_pmt_top @ V[:,d] for each direction
    img_scores_all = X_val_img_top @ U  # [N_val, r]
    pmt_scores_all = X_val_pmt_top @ V  # [N_val, r]
    dir_scores_all = img_scores_all * pmt_scores_all  # [N_val, r]

    # ── Pseudo pairs (train-split rows) ───────────────────────────────────
    # IVL dev_seen has no cross-label pairs; use pair_analysis_splits (default: train)
    # to get meaningful pseudo-pair deltas.  Extract separate img/pmt feature
    # matrices for those rows, compute per-direction scores, then build pairs.
    pair_rows = _manifest_rows(cache_dir, cfg.pair_analysis_splits, "Hateful_Memes_Dataset")
    logger.info("Pair-analysis rows: %d (splits=%s)", len(pair_rows), cfg.pair_analysis_splits)

    if pair_rows and cfg.fhm_confounder_metadata_path:
        logger.info("Extracting pair-analysis image features …")
        pair_keys_img, X_pair_img = _extract_modality_features(
            pair_rows, layer_specs, crosscoder, img_tok_sel, cfg, device)
        logger.info("Extracting pair-analysis prompt features …")
        pair_keys_pmt, X_pair_pmt = _extract_modality_features(
            pair_rows, layer_specs, crosscoder, pmt_tok_sel, cfg, device)
        assert pair_keys_img == pair_keys_pmt, "Pair-analysis key lists misaligned"
        pair_keys = pair_keys_img
        # Scale with same scalers fitted on train
        X_pair_img_top = scaler_img.transform(X_pair_img)[:, feat_idx_arr].toarray().astype(np.float32)
        X_pair_pmt_top = scaler_pmt.transform(X_pair_pmt)[:, feat_idx_arr].toarray().astype(np.float32)
        pair_img_scores_all = X_pair_img_top @ U  # [N_pair, r]
        pair_pmt_scores_all = X_pair_pmt_top @ V  # [N_pair, r]
        pair_dir_scores_all = pair_img_scores_all * pair_pmt_scores_all  # [N_pair, r]
        pair_key_to_idx = {k: i for i, k in enumerate(pair_keys)}
        pseudo_pairs = _build_fhm_pseudo_pairs(
            cfg.fhm_confounder_metadata_path, set(pair_keys))
    else:
        pair_dir_scores_all = dir_scores_all  # fallback to val scores
        pair_key_to_idx = key_to_val_idx
        pseudo_pairs = _build_fhm_pseudo_pairs(
            cfg.fhm_confounder_metadata_path, set(val_keys))

    logger.info("Pseudo pairs: %s",
                {src: len(pairs) for src, pairs in pseudo_pairs.items()})

    # ── Build cards ───────────────────────────────────────────────────────────
    cards = []
    for d in range(r):
        logger.info("Building card for direction %d/%d …", d + 1, r)
        card = _build_card(
            d=d,
            U=U, V=V,
            feat_indices=feat_indices,
            img_scores=img_scores_all[:, d],
            pmt_scores=pmt_scores_all[:, d],
            dir_scores=dir_scores_all[:, d],
            full_logits=full_logits_np,
            bil_scale=bil_scale,
            threshold=threshold,
            val_keys=val_keys,
            y_val=y_val,
            meta_by_key=val_manifest_by_key,
            pseudo_pairs=pseudo_pairs,
            key_to_val_idx=pair_key_to_idx,
            pair_dir_scores=pair_dir_scores_all[:, d],
            descriptions=descriptions,
            audit_scores=audit_sc,
            top_k_features=cfg.top_k_features,
            top_k_pairs=cfg.top_k_pairs,
            top_k_examples=cfg.top_k_examples,
            n_counterexamples=cfg.n_counterexamples,
            global_doc_freq=global_doc_freq,
            global_n_docs=global_n_docs,
        )
        card["total_directions"] = r
        cards.append(card)

    # Sort by strength descending
    cards.sort(key=lambda c: -c["strength"])
    for rank, card in enumerate(cards):
        card["strength_rank"] = rank + 1
        card["is_primary_card"] = rank < cfg.top_n_cards

    # ── Write outputs ─────────────────────────────────────────────────────────
    (out_dir / "interaction_cards.json").write_text(json.dumps(cards, indent=2))
    logger.info("Wrote %d interaction cards to %s", len(cards), out_dir)

    # Human-readable text report -- only the top_n_cards strongest directions
    # (the "interaction cards" deliverable). All directions remain available in
    # interaction_cards.json / direction_summary.json for completeness.
    primary_cards = [c for c in cards if c["is_primary_card"]]
    report_lines = [
        "FHM Bilinear Interaction Cards",
        f"Checkpoint: {ckpt_dir}",
        f"Rank r={r}, K={K} confounder features, scale={bil_scale:.4f}",
        f"Threshold: {threshold:.3f} (from calib split)",
        f"Val samples: {len(val_keys)}, Pseudo pairs: {pseudo_pairs}",
        f"Showing top {len(primary_cards)}/{len(cards)} directions by strength "
        f"(all {len(cards)} directions are in interaction_cards.json)",
        "",
    ]
    for card in sorted(primary_cards, key=lambda c: c["strength_rank"]):
        report_lines.append(_format_card_txt(card))

    (out_dir / "interaction_report.txt").write_text("\n".join(report_lines))
    logger.info("Wrote interaction_report.txt (%d primary cards)", len(primary_cards))

    # Flat JSONL of top examples for easy browsing
    with (out_dir / "top_examples.jsonl").open("w") as fd:
        for card in cards:
            for ex in card["top_activating_examples"]:
                fd.write(json.dumps({
                    "direction": card["direction"],
                    "direction_1indexed": card["direction_1indexed"],
                    "strength_rank": card["strength_rank"],
                    "candidate_label": card["candidate_semantic_label"],
                    **ex,
                }) + "\n")
    logger.info("Wrote top_examples.jsonl")

    # Direction summary table
    summary = []
    for card in sorted(cards, key=lambda c: c["direction"]):
        ab = card["ablation"]
        ps = card["pseudo_pair_deltas"]
        summary.append({
            "direction": card["direction_1indexed"],
            "strength": card["strength"],
            "strength_rank": card["strength_rank"],
            "is_primary_card": card["is_primary_card"],
            "candidate_label": card["candidate_semantic_label"],
            "ablation_delta_f1": ab["delta_macro_f1"],
            "n_preds_changed": ab["n_predictions_changed"],
            "pseudo_img_pos_margin": (ps.get("pseudo_img") or {}).get("positive_margin_rate"),
            "pseudo_txt_pos_margin": (ps.get("pseudo_text") or {}).get("positive_margin_rate"),
            "top_u_feature": (card["u_features_image_side"][0]["feature_idx"]
                              if card["u_features_image_side"] else None),
            "top_v_feature": (card["v_features_prompt_side"][0]["feature_idx"]
                              if card["v_features_prompt_side"] else None),
        })
    (out_dir / "direction_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote direction_summary.json")

    # Print summary table to stdout
    print("\n" + "=" * 80)
    print("Bilinear Interaction Direction Summary")
    print(f"r={r}, K={K}, scale={bil_scale:.4f}, threshold={threshold:.3f}")
    print(f"Primary cards (top {cfg.top_n_cards} by strength) marked with *")
    print(f"{'Dir':>4}  {'Rank':>4}  {'Strength':>9}  {'AblΔF1':>8}  {'ΔPreds':>7}  "
          f"{'img_pos_mg':>10}  {'txt_pos_mg':>10}  Candidate label")
    print("-" * 100)
    for s in sorted(summary, key=lambda x: x["direction"]):
        pm_i = f"{s['pseudo_img_pos_margin']:.3f}" if s["pseudo_img_pos_margin"] is not None else "  —  "
        pm_t = f"{s['pseudo_txt_pos_margin']:.3f}" if s["pseudo_txt_pos_margin"] is not None else "  —  "
        marker = "*" if s["is_primary_card"] else " "
        print(f"{marker} {s['direction']:>2}  {s['strength_rank']:>4}  {s['strength']:>9.4f}  "
              f"{s['ablation_delta_f1']:>+8.4f}  {s['n_preds_changed']:>7}  "
              f"{pm_i:>10}  {pm_t:>10}  {s['candidate_label'][:50]}")
    print("=" * 80)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(InteractionCardConfig,
                argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
