

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.analyze_fhm_confounder_features import _build_pairs
from gemma3_experiments.launch.residual.probe_fhm_crosscoder import (
    DenseLayerSpec,
    DenseShardReader,
    GemmaCrosscoder,
    _aggregate_acts,
    _common_keys,
    _manifest_key,
    _role_mask,
)
from gemma3_experiments.launch.residual.probe_validate import (
    _build_fhm_metadata_hard_negative_pairs,
    _calibrate_threshold,
    _load_fhm_confounder_metadata,
    _normalise_fhm_id,
    _normalise_group_value,
    _per_class_metrics,
    _read_jsonl,
    _row_gold_label,
    _row_sample_key,
)
from gemma3_experiments.residual_sae import resolve_visible_device

logger = logging.getLogger(__name__)

TASK_KEY = "Hateful_Memes_Dataset::hateful_memes_binary::label"
FHM_LABELS = ["hateful", "not hateful"]
_L2I = {"hateful": 1, "not hateful": 0}


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CrosscoderPairwiseConfig:
    # Crosscoder / SAE
    crosscoder_repo_id: str = "google/gemma-scope-2-4b-it"
    crosscoder_hookpoint: str = "crosscoder/layer_9_17_22_29_width_65k_l0_medium"

    # Dense cache dirs (reuse layout from probe_fhm_crosscoder.py)
    layer9_name: str = "layer9"
    layer9_train_dense_cache_dir: str = ""
    layer9_val_dense_cache_dir: str = ""
    layer17_name: str = "layer17"
    layer17_train_dense_cache_dir: str = ""
    layer17_val_dense_cache_dir: str = ""
    layer22_name: str = "layer22"
    layer22_train_dense_cache_dir: str = ""
    layer22_val_dense_cache_dir: str = ""
    layer29_name: str = "layer29"
    layer29_train_dense_cache_dir: str = ""
    layer29_val_dense_cache_dir: str = ""

    # IO
    baseline_predictions_path: str = ""
    fhm_confounder_metadata_path: str = ""
    output_dir: str = ""
    eval_splits: list[str] = field(default_factory=lambda: ["validation", "val", "dev"])
    # Save model state_dict + bilinear feature indices after training so that
    # analyze_bilinear_factors.py can inspect the U/V factor columns.
    save_checkpoint: bool = True

    # Feature extraction
    token_selection: str = "all"
    aggregation: str = "max"
    batch_size: int = 64
    chunk_size: int = 64
    shard_cache_size: int = 16

    # Pair sources for hinge + gate
    pair_sources: list[str] = field(default_factory=lambda: ["pseudo_img", "pseudo_text"])

    # Architecture
    # hidden_size=128 keeps each head at ~8M params for 65k features, matching
    # what worked in the old residual-SAE gated pairwise (20k features × 256 = 5M).
    # Larger hidden_size (256) with 65k features gives 16.8M params per head —
    # too many for 7938 FHM samples to train before BCE collapse wins.
    hidden_size: int = 128      # per-head hidden dim
    gate_hidden_size: int = 32  # gate MLP hidden dim
    dropout: float = 0.2

    # Optional bilinear term over top confounder features
    use_bilinear: bool = False
    bilinear_top_k: int = 128   # number of confounder features per modality
    # Path to confounder_feature_top_features.jsonl from the crosscoder audit.
    # Features are ranked by sign_consistency × effect_size and deduped across
    # pair sources; top bilinear_top_k are used as the bilinear column indices.
    bilinear_audit_path: str = ""
    bilinear_audit_token_selection: str = "prompt_and_image"  # which tok-sel to rank from
    # Separate token selections for the two bilinear slots.
    bilinear_image_token_selection: str = "image"
    bilinear_prompt_token_selection: str = "prompt"
    # Low-rank factorization of W_bil: W = U @ V.T, U,V ∈ R^(K×r).
    # 0 = full rank (K×K, current behaviour).  r>0 is interpretable minimum rank.
    bilinear_rank: int = 0
    # Triple cross-modal bilinear: adds I×G and P×G terms on top of I×P.
    # Requires a generated-token cache at the same cache dirs.
    use_triple_bilinear: bool = False
    bilinear_generated_token_selection: str = "generated"

    # Training
    n_epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-3
    patience: int = 20
    eval_every: int = 10

    # Loss weights
    margin: float = 0.2
    alpha: float = 0.80         # BCE weight
    pos_weight: float = 2.0     # amplify hateful BCE gradient to prevent collapse
    add_delta_rows: bool = True  # pseudo-HN delta rows complement pairwise hinge

    # Calibration split
    # Reserve calibration_frac of validation for threshold tuning; report on rest.
    calibration_frac: float = 0.20
    calibration_seed: int = 42

    # Training seed — controls torch weight init + numpy shuffles before model
    # construction so that seed variance across runs is reproducible.
    # calibration_seed is kept separate so the calib/held-out split is always
    # the same regardless of which training seed is used.
    training_seed: int = 0

    device: str = "cuda:0"


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class CrosscoderPairwiseReadout(nn.Module):
    """Source-specific pairwise readout over crosscoder feature vectors.

    img_head  — specialises on pseudo_img pair structure (same image, diff text)
    text_head — specialises on pseudo_text pair structure (same text, diff image)
    gate      — feature-based scalar; trained by BCE gradient, no explicit supervision
    score(x)  = gate(x) · img_head(x) + (1−gate(x)) · text_head(x)
    """

    def __init__(self, n_features: int, hidden_size: int, gate_hidden: int,
                    dropout: float = 0.2, bilinear_k: int = 0,
                    bilinear_rank: int = 0, triple_bilinear: bool = False):
        super().__init__()
        self.n_features = n_features
        self.bilinear_k = bilinear_k
        self.bilinear_rank = bilinear_rank
        self.triple_bilinear = triple_bilinear

        self.img_head = nn.Sequential(
            nn.Linear(n_features, hidden_size), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.text_head = nn.Sequential(
            nn.Linear(n_features, hidden_size), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        # Lightweight feature-based gate — small to prevent gate overfitting
        self.gate = nn.Sequential(
            nn.Linear(n_features, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, 1), nn.Sigmoid(),
        )
        # Optional bilinear interaction over top confounder features
        if bilinear_k > 0:
            r = bilinear_rank if bilinear_rank > 0 else bilinear_k
            if bilinear_rank > 0:
                # Low-rank: W_bil = U @ V.T,  U,V ∈ R^(K×r)
                self.bil_U = nn.Parameter(torch.zeros(bilinear_k, r))
                self.bil_V = nn.Parameter(torch.zeros(bilinear_k, r))
                nn.init.xavier_uniform_(self.bil_U)
                nn.init.xavier_uniform_(self.bil_V)
                self.W_bil = None  # unused in low-rank mode
            else:
                # Full-rank K×K matrix
                self.W_bil = nn.Parameter(torch.zeros(bilinear_k, bilinear_k))
                nn.init.xavier_uniform_(self.W_bil)
                self.bil_U = self.bil_V = None
            self.bil_scale = nn.Parameter(torch.tensor(0.1))
            # Triple bilinear: P×G and I×G cross terms (same rank as I×P)
            if triple_bilinear:
                if bilinear_rank > 0:
                    self.bil_PG_U = nn.Parameter(torch.zeros(bilinear_k, r))
                    self.bil_PG_V = nn.Parameter(torch.zeros(bilinear_k, r))
                    self.bil_IG_U = nn.Parameter(torch.zeros(bilinear_k, r))
                    self.bil_IG_V = nn.Parameter(torch.zeros(bilinear_k, r))
                    nn.init.xavier_uniform_(self.bil_PG_U)
                    nn.init.xavier_uniform_(self.bil_PG_V)
                    nn.init.xavier_uniform_(self.bil_IG_U)
                    nn.init.xavier_uniform_(self.bil_IG_V)
                else:
                    self.W_bil_PG = nn.Parameter(torch.zeros(bilinear_k, bilinear_k))
                    self.W_bil_IG = nn.Parameter(torch.zeros(bilinear_k, bilinear_k))
                    nn.init.xavier_uniform_(self.W_bil_PG)
                    nn.init.xavier_uniform_(self.W_bil_IG)
                self.bil_PG_scale = nn.Parameter(torch.tensor(0.1))
                self.bil_IG_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.W_bil = self.bil_U = self.bil_V = None

    def _score_head(self, head: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        return head(x).squeeze(-1)  # [B]

    def forward(self, x: torch.Tensor,
        x_img_top: torch.Tensor | None = None,
        x_pmt_top: torch.Tensor | None = None,
        x_gen_top: torch.Tensor | None = None) -> torch.Tensor:
        g = self.gate(x)  # [B, 1]
        s_img  = self._score_head(self.img_head,  x)  # [B]
        s_text = self._score_head(self.text_head, x)  # [B]
        combined = (g.squeeze(-1) * s_img + (1 - g.squeeze(-1)) * s_text)
        if x_img_top is not None and x_pmt_top is not None:
            if self.bilinear_rank > 0:
                # Low-rank I×P:  (I @ U) ⊙ (P @ V)  →  sum over r dims
                bil = (x_img_top @ self.bil_U) * (x_pmt_top @ self.bil_V)  # [B, r]
            elif self.W_bil is not None:
                bil = (x_img_top @ self.W_bil) * x_pmt_top  # [B, k]
            else:
                bil = None
            if bil is not None:
                combined = combined + self.bil_scale * bil.sum(dim=-1)
            # Triple bilinear: P×G and I×G
            if self.triple_bilinear and x_gen_top is not None:
                if self.bilinear_rank > 0:
                    bil_PG = (x_pmt_top @ self.bil_PG_U) * (x_gen_top @ self.bil_PG_V)
                    bil_IG = (x_img_top @ self.bil_IG_U) * (x_gen_top @ self.bil_IG_V)
                else:
                    bil_PG = (x_pmt_top @ self.W_bil_PG) * x_gen_top
                    bil_IG = (x_img_top @ self.W_bil_IG) * x_gen_top
                combined = (combined
                            + self.bil_PG_scale * bil_PG.sum(dim=-1)
                            + self.bil_IG_scale * bil_IG.sum(dim=-1))
        return combined

    def img_score(self, x: torch.Tensor) -> torch.Tensor:
        return self._score_head(self.img_head, x)

    def text_score(self, x: torch.Tensor) -> torch.Tensor:
        return self._score_head(self.text_head, x)


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction (reuses DenseShardReader from probe_fhm_crosscoder)
# ──────────────────────────────────────────────────────────────────────────────

def _layer_specs(cfg: CrosscoderPairwiseConfig) -> list[DenseLayerSpec]:
    candidates = [
        (cfg.layer9_name,  cfg.layer9_train_dense_cache_dir,  cfg.layer9_val_dense_cache_dir),
        (cfg.layer17_name, cfg.layer17_train_dense_cache_dir, cfg.layer17_val_dense_cache_dir),
        (cfg.layer22_name, cfg.layer22_train_dense_cache_dir, cfg.layer22_val_dense_cache_dir),
        (cfg.layer29_name, cfg.layer29_train_dense_cache_dir, cfg.layer29_val_dense_cache_dir),
    ]
    return [DenseLayerSpec(n, tr, vl) for n, tr, vl in candidates if tr and vl]


def _load_fhm_rows(cache_dir: Path, split_names: set[str]) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(cache_dir / "manifest.jsonl"):
        if row.get("dataset_name") != "Hateful_Memes_Dataset":
            continue
        if row.get("task_name") != "hateful_memes_binary":
            continue
        if row.get("split") not in split_names:
            continue
        if _row_gold_label(row) not in set(FHM_LABELS):
            continue
        row = dict(row)
        row["_cache_dir"] = str(cache_dir)
        rows.append(row)
    return rows


def _extract_features_cc(
    rows_by_layer: list[list[dict[str, Any]]],
    key_lists: list[list[str]],
    layer_specs: list[DenseLayerSpec],
    crosscoder: GemmaCrosscoder,
    token_selection: str,
    aggregation: str,
    batch_size: int,
    chunk_size: int,
    shard_cache_size: int,
    device: str,
) -> tuple[list[str], Any]:
    keys = _common_keys(key_lists)
    lookup = [{_manifest_key(r): r for r in rows} for rows in rows_by_layer]
    reader = DenseShardReader(layer_specs, shard_cache_size, token_selection)
    feature_rows = []
    for start in range(0, len(keys), batch_size):
        for key in keys[start : start + batch_size]:
            sample_rows = [lu[key] for lu in lookup]
            hidden_tld, roles = reader.get_sample(sample_rows)
            mask = _role_mask(roles, token_selection)
            sel = hidden_tld[mask]
            chunks = [crosscoder.encode(sel[s : s + chunk_size]) for s in range(0, sel.shape[0], chunk_size)]
            vec = _aggregate_acts(torch.cat(chunks), aggregation) if chunks else np.zeros(crosscoder.num_latents, dtype=np.float32)
            feature_rows.append(scipy.sparse.csr_matrix(vec.reshape(1, -1)))
        if (start + batch_size) % 512 == 0 or start + batch_size >= len(keys):
            logger.info("  features: %d / %d", min(start + batch_size, len(keys)), len(keys))
    return keys, scipy.sparse.vstack(feature_rows, format="csr", dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Pair building and index utilities
# ──────────────────────────────────────────────────────────────────────────────

def _load_bilinear_feature_indices(
    jsonl_path: str, top_k: int, token_selection: str
) -> np.ndarray:
    """Return the top_k most-scored confounder feature indices for the given token selection.

    Scores from both pseudo_img and pseudo_text sources are merged; for duplicate
    feature_idx the higher score is kept. Features are then sorted descending and
    the top top_k indices are returned.
    """
    scores: dict[int, float] = {}
    for row in _read_jsonl(Path(jsonl_path)):
        if row.get("token_selection") != token_selection:
            continue
        feat_idx = int(row["feature_idx"])
        score = float(row.get("score", 0.0))
        if feat_idx not in scores or scores[feat_idx] < score:
            scores[feat_idx] = score
    if not scores:
        raise ValueError(
            f"No features found for token_selection={token_selection!r} in {jsonl_path}. "
            "Check --bilinear_audit_path and --bilinear_audit_token_selection."
        )
    sorted_feats = sorted(scores.items(), key=lambda x: -x[1])
    return np.array([f for f, _ in sorted_feats[:top_k]], dtype=np.int64)


def _pair_index_arrays(
    pairs: list[tuple[str, str]],
    key_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (hateful_indices, benign_indices) for all filterable pairs."""
    hi, bi = [], []
    for hk, bk in pairs:
        if hk in key_to_idx and bk in key_to_idx:
            hi.append(key_to_idx[hk])
            bi.append(key_to_idx[bk])
    return np.array(hi, dtype=np.int64), np.array(bi, dtype=np.int64)


def _pair_metrics_local(
    pairs: list[tuple[str, str]],
    pred_by_key: dict[str, str],
    proba_by_key: dict[str, dict[str, float]],
) -> dict[str, Any]:
    rows = []
    for hk, bk in pairs:
        if hk not in pred_by_key or bk not in pred_by_key:
            continue
        hp = proba_by_key[hk].get("hateful", 0.0)
        bp = proba_by_key[bk].get("hateful", 0.0)
        rows.append({
            "hateful_key": hk, "benign_key": bk,
            "both_correct": pred_by_key[hk] == "hateful" and pred_by_key[bk] == "not hateful",
            "separated": pred_by_key[hk] != pred_by_key[bk],
            "positive_margin": hp > bp,
            "prob_gap": hp - bp,
        })
    if not rows:
        return {"n_pairs": 0}
    return {
        "n_pairs": len(rows),
        "both_correct_rate": round(float(np.mean([r["both_correct"] for r in rows])), 6),
        "separated_rate": round(float(np.mean([r["separated"] for r in rows])), 6),
        "positive_margin_rate": round(float(np.mean([r["positive_margin"] for r in rows])), 6),
        "median_prob_gap": round(float(np.median([r["prob_gap"] for r in rows])), 6),
        "rows": rows,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def _train(
    model: CrosscoderPairwiseReadout,
    X_all: torch.Tensor,           # [N_train, F] dense float32 on CPU
    y_all: torch.Tensor,           # [N_train]  float32 (0/1)
    img_h_idx: torch.Tensor,
    img_b_idx: torch.Tensor,
    txt_h_idx: torch.Tensor,
    txt_b_idx: torch.Tensor,
    sample_weights: torch.Tensor | None,
    cfg: CrosscoderPairwiseConfig,
    device: str,
    val_X: torch.Tensor,
    val_y: list[str],
    # Optional bilinear tensors — None when use_bilinear=False
    X_img_top: torch.Tensor | None = None,   # [N_train, K]
    X_pmt_top: torch.Tensor | None = None,   # [N_train, K]
    X_gen_top: torch.Tensor | None = None,   # [N_train, K]  (triple bilinear)
    val_X_img_top: torch.Tensor | None = None,  # [N_val_calib, K]
    val_X_pmt_top: torch.Tensor | None = None,
    val_X_gen_top: torch.Tensor | None = None,
) -> tuple[int, float]:
    """Full-batch training. Returns (best_epoch, best_val_f1)."""
    pos_wt = torch.tensor(cfg.pos_weight, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    alpha = cfg.alpha
    pair_weight = 1.0 - alpha

    best_f1, best_epoch, no_improve = 0.0, 0, 0
    best_state = None

    X_dev = X_all.to(device)
    img_h = img_h_idx.to(device); img_b = img_b_idx.to(device)
    txt_h = txt_h_idx.to(device); txt_b = txt_b_idx.to(device)
    y_dev = y_all.to(device)
    if sample_weights is not None:
        sw_dev = sample_weights.to(device)
    # Bilinear tensors on device (or None)
    bil_img = X_img_top.to(device) if X_img_top is not None else None
    bil_pmt = X_pmt_top.to(device) if X_pmt_top is not None else None
    bil_gen = X_gen_top.to(device) if X_gen_top is not None else None
    val_bil_img = val_X_img_top.to(device) if val_X_img_top is not None else None
    val_bil_pmt = val_X_pmt_top.to(device) if val_X_pmt_top is not None else None
    val_bil_gen = val_X_gen_top.to(device) if val_X_gen_top is not None else None

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        scores = model(X_dev, bil_img, bil_pmt, bil_gen)  # [N]
        bce = F.binary_cross_entropy_with_logits(scores, y_dev, pos_weight=pos_wt,
                                                  weight=sw_dev if sample_weights is not None else None)
        loss = alpha * bce
        if img_h.numel() > 0 and "pseudo_img" in cfg.pair_sources:
            hs = model.img_score(X_dev[img_h])
            bs = model.img_score(X_dev[img_b])
            loss = loss + pair_weight * F.relu(bs - hs + cfg.margin).mean()
        if txt_h.numel() > 0 and "pseudo_text" in cfg.pair_sources:
            hs = model.text_score(X_dev[txt_h])
            bs = model.text_score(X_dev[txt_b])
            loss = loss + pair_weight * F.relu(bs - hs + cfg.margin).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % cfg.eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_scores = torch.sigmoid(model(val_X.to(device), val_bil_img, val_bil_pmt, val_bil_gen)).cpu().numpy()
            val_pred = ["hateful" if s >= 0.5 else "not hateful" for s in val_scores]
            m = _per_class_metrics(val_y, val_pred)
            f1 = m["macro_f1"]
            if f1 > best_f1:
                best_f1 = f1; best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if epoch % 50 == 0:
                logger.info("  epoch %d  loss=%.4f  val_f1=%.4f  best=%.4f@%d",
                            epoch, loss.item(), f1, best_f1, best_epoch)
            if no_improve >= cfg.patience // cfg.eval_every:
                logger.info("  Early stop at epoch %d", epoch)
                break

    if best_state:
        model.load_state_dict(best_state)
    return best_epoch, best_f1


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_crosscoder_pairwise(cfg: CrosscoderPairwiseConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(cfg.fhm_confounder_metadata_path)
    baseline_path = Path(cfg.baseline_predictions_path) if cfg.baseline_predictions_path else None
    assert metadata_path.exists(), f"FHM metadata missing: {metadata_path}"

    layer_specs = _layer_specs(cfg)
    assert len(layer_specs) >= 1, "Need at least one dense layer cache."

    # ── Load rows ──────────────────────────────────────────────────────────────
    train_rows_by_layer, val_rows_by_layer = [], []
    train_key_lists, val_key_lists = [], []
    eval_set = set(cfg.eval_splits)
    for spec in layer_specs:
        tr = _load_fhm_rows(Path(spec.train_dense_cache_dir), {"train"})
        vl = _load_fhm_rows(Path(spec.val_dense_cache_dir), eval_set)
        train_rows_by_layer.append(tr)
        val_rows_by_layer.append(vl)
        train_key_lists.append([_manifest_key(r) for r in tr])
        val_key_lists.append([_manifest_key(r) for r in vl])
        logger.info("%s: train=%d val=%d", spec.name, len(tr), len(vl))

    # ── Load crosscoder ────────────────────────────────────────────────────────
    first_shard = Path(train_rows_by_layer[0][0]["_cache_dir"]) / train_rows_by_layer[0][0]["shard_file"]
    with safe_open(str(first_shard), framework="pt", device="cpu") as fd:
        d_model = int(fd.get_tensor("hidden").shape[1])
    device = resolve_visible_device(cfg.device)
    crosscoder = GemmaCrosscoder.load_from_hub(
        cfg.crosscoder_repo_id, cfg.crosscoder_hookpoint,
        n_layers=len(layer_specs), d_model=d_model, device=device,
    )
    logger.info("Crosscoder: num_latents=%d hookpoint=%s", crosscoder.num_latents, cfg.crosscoder_hookpoint)

    # ── Extract features ───────────────────────────────────────────────────────
    kwargs = dict(aggregation=cfg.aggregation, batch_size=cfg.batch_size,
                  chunk_size=cfg.chunk_size, shard_cache_size=cfg.shard_cache_size, device=device)
    logger.info("Extracting train features (token_selection=%s)...", cfg.token_selection)
    train_keys, X_train_base = _extract_features_cc(
        train_rows_by_layer, train_key_lists, layer_specs, crosscoder,
        cfg.token_selection, **kwargs)
    logger.info("Extracting val features...")
    val_keys, X_val_base = _extract_features_cc(
        val_rows_by_layer, val_key_lists, layer_specs, crosscoder,
        cfg.token_selection, **kwargs)

    # ── Build IVL pairs ────────────────────────────────────────────────────────
    metadata = _load_fhm_confounder_metadata(metadata_path)
    train_row_by_key = {_manifest_key(r): r for r in train_rows_by_layer[0]}
    val_row_by_key   = {_manifest_key(r): r for r in val_rows_by_layer[0]}
    train_key_to_idx = {k: i for i, k in enumerate(train_keys)}
    val_key_to_idx   = {k: i for i, k in enumerate(val_keys)}

    train_pairs: dict[str, list[tuple[str, str]]] = {}
    val_pairs:   dict[str, list[tuple[str, str]]] = {}
    for src in cfg.pair_sources:
        train_pairs[src] = _build_pairs(
            [train_row_by_key[k] for k in train_keys], metadata, set(train_keys), src)
        val_pairs[src] = _build_pairs(
            [val_row_by_key[k] for k in val_keys], metadata, set(val_keys), src)
        logger.info("%s: train_pairs=%d val_pairs=%d", src, len(train_pairs[src]), len(val_pairs[src]))

    # ── Pseudo-HN delta rows ────────────────────────────────────────────────────
    X_train = X_train_base
    y_train = [_row_gold_label(train_row_by_key[k]) for k in train_keys]
    delta_stats = None
    if cfg.add_delta_rows:
        delta_csr, delta_labels, delta_stats = _build_fhm_metadata_hard_negative_pairs(
            [train_row_by_key[k] for k in train_keys],
            X_train_base, train_key_to_idx, metadata_path, cfg.pair_sources,
        )
        if delta_csr.shape[0]:
            X_train = scipy.sparse.vstack([X_train_base, delta_csr], format="csr")
            y_train = list(y_train) + delta_labels
            logger.info("Delta rows: %d real → %d after delta", len(train_keys), X_train.shape[0])

    from sklearn.preprocessing import MaxAbsScaler
    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # CSR in, CSR out (sparse-safe)
    X_val_scaled   = scaler.transform(X_val_base)

    # ── Convert to dense tensors ────────────────────────────────────────────────
    logger.info("Converting to dense tensors: train=%s val=%s features=%d",
                X_train.shape, X_val_base.shape, crosscoder.num_latents)
    X_tr_dense  = torch.from_numpy(X_train_scaled.toarray().astype(np.float32))
    X_val_dense = torch.from_numpy(X_val_scaled.toarray().astype(np.float32))
    y_tr_float  = torch.tensor([_L2I.get(lbl, 0) for lbl in y_train], dtype=torch.float32)

    # Pair index arrays for hinge loss (only over real rows, not delta rows)
    n_real = len(train_keys)
    img_h_np, img_b_np = _pair_index_arrays(
        train_pairs.get("pseudo_img", []), train_key_to_idx)
    txt_h_np, txt_b_np = _pair_index_arrays(
        train_pairs.get("pseudo_text", []), train_key_to_idx)
    img_h_t = torch.from_numpy(img_h_np); img_b_t = torch.from_numpy(img_b_np)
    txt_h_t = torch.from_numpy(txt_h_np); txt_b_t = torch.from_numpy(txt_b_np)
    logger.info("Hinge pairs: pseudo_img=%d  pseudo_text=%d", len(img_h_np), len(txt_h_np))

    # ── Calibration split ────────────────────────────────────────────────────────
    rng = np.random.default_rng(cfg.calibration_seed)
    n_val = len(val_keys)
    n_calib = max(1, round(n_val * cfg.calibration_frac))
    calib_idx = rng.choice(n_val, n_calib, replace=False)
    held_idx  = np.setdiff1d(np.arange(n_val), calib_idx)
    logger.info("Val split: calib=%d held-out=%d (frac=%.2f)", n_calib, len(held_idx), cfg.calibration_frac)
    y_val_all = [_row_gold_label(val_row_by_key[k]) for k in val_keys]
    X_val_calib  = X_val_dense[calib_idx]
    X_val_held   = X_val_dense[held_idx]
    y_val_calib  = [y_val_all[i] for i in calib_idx]
    y_val_held   = [y_val_all[i] for i in held_idx]
    val_keys_held = [val_keys[i] for i in held_idx]

    # ── Bilinear feature extraction (when use_bilinear=True) ─────────────────────
    X_tr_img_top = X_tr_pmt_top = None
    X_val_img_top = X_val_pmt_top = None
    bilinear_top_k_actual = 0
    if cfg.use_bilinear:
        assert cfg.bilinear_audit_path, "--bilinear_audit_path required when --use_bilinear true"
        top_k_feats = _load_bilinear_feature_indices(
            cfg.bilinear_audit_path, cfg.bilinear_top_k, cfg.bilinear_audit_token_selection
        )
        bilinear_top_k_actual = len(top_k_feats)
        logger.info("Bilinear: %d features (from %s, tok_sel=%s)",
                    bilinear_top_k_actual, cfg.bilinear_audit_path, cfg.bilinear_audit_token_selection)

        def _bil_extract(tok_sel: str, is_train: bool) -> np.ndarray:
            rows_bl = train_rows_by_layer if is_train else val_rows_by_layer
            keys_bl = train_key_lists    if is_train else val_key_lists
            _, X_bl = _extract_features_cc(
                rows_bl, keys_bl, layer_specs, crosscoder, tok_sel, **kwargs)
            sc_bl = MaxAbsScaler().fit_transform(X_bl) if is_train else X_bl
            # For val: fit scaler on train then transform val
            return sc_bl[:, top_k_feats].toarray().astype(np.float32)

        # Fit scalers on train, transform val (no leakage)
        img_scaler_bil = MaxAbsScaler()
        pmt_scaler_bil = MaxAbsScaler()

        logger.info("Extracting bilinear image features (token_sel=%s)...", cfg.bilinear_image_token_selection)
        _, X_img_base = _extract_features_cc(
            train_rows_by_layer, train_key_lists, layer_specs, crosscoder,
            cfg.bilinear_image_token_selection, **kwargs)
        _, X_val_img_base = _extract_features_cc(
            val_rows_by_layer, val_key_lists, layer_specs, crosscoder,
            cfg.bilinear_image_token_selection, **kwargs)
        img_tr_scaled = img_scaler_bil.fit_transform(X_img_base)[:, top_k_feats].toarray().astype(np.float32)
        img_val_scaled = img_scaler_bil.transform(X_val_img_base)[:, top_k_feats].toarray().astype(np.float32)

        logger.info("Extracting bilinear prompt features (token_sel=%s)...", cfg.bilinear_prompt_token_selection)
        _, X_pmt_base = _extract_features_cc(
            train_rows_by_layer, train_key_lists, layer_specs, crosscoder,
            cfg.bilinear_prompt_token_selection, **kwargs)
        _, X_val_pmt_base = _extract_features_cc(
            val_rows_by_layer, val_key_lists, layer_specs, crosscoder,
            cfg.bilinear_prompt_token_selection, **kwargs)
        pmt_tr_scaled = pmt_scaler_bil.fit_transform(X_pmt_base)[:, top_k_feats].toarray().astype(np.float32)
        pmt_val_scaled = pmt_scaler_bil.transform(X_val_pmt_base)[:, top_k_feats].toarray().astype(np.float32)

        # Pad delta rows with zeros (they are feature diffs, not meaningful for bilinear)
        n_delta = X_tr_dense.shape[0] - len(train_keys)
        if n_delta > 0:
            img_tr_scaled = np.vstack([img_tr_scaled, np.zeros((n_delta, bilinear_top_k_actual), dtype=np.float32)])
            pmt_tr_scaled = np.vstack([pmt_tr_scaled, np.zeros((n_delta, bilinear_top_k_actual), dtype=np.float32)])

        X_tr_img_top  = torch.from_numpy(img_tr_scaled)
        X_tr_pmt_top  = torch.from_numpy(pmt_tr_scaled)
        X_val_img_top = torch.from_numpy(img_val_scaled)
        X_val_pmt_top = torch.from_numpy(pmt_val_scaled)
        logger.info("Bilinear tensors: train=%s val=%s", tuple(X_tr_img_top.shape), tuple(X_val_img_top.shape))

        # Optional triple bilinear: extract generated-token features for P×G and I×G
        X_tr_gen_top = X_val_gen_top = None
        if cfg.use_triple_bilinear:
            gen_scaler_bil = MaxAbsScaler()
            logger.info("Extracting bilinear generated features (token_sel=%s)...",
                        cfg.bilinear_generated_token_selection)
            _, X_gen_base = _extract_features_cc(
                train_rows_by_layer, train_key_lists, layer_specs, crosscoder,
                cfg.bilinear_generated_token_selection, **kwargs)
            _, X_val_gen_base = _extract_features_cc(
                val_rows_by_layer, val_key_lists, layer_specs, crosscoder,
                cfg.bilinear_generated_token_selection, **kwargs)
            gen_tr_scaled = gen_scaler_bil.fit_transform(X_gen_base)[:, top_k_feats].toarray().astype(np.float32)
            gen_val_scaled = gen_scaler_bil.transform(X_val_gen_base)[:, top_k_feats].toarray().astype(np.float32)
            if n_delta > 0:
                gen_tr_scaled = np.vstack([gen_tr_scaled, np.zeros((n_delta, bilinear_top_k_actual), dtype=np.float32)])
            X_tr_gen_top  = torch.from_numpy(gen_tr_scaled)
            X_val_gen_top = torch.from_numpy(gen_val_scaled)
            logger.info("Triple bilinear gen tensors: train=%s val=%s",
                        tuple(X_tr_gen_top.shape), tuple(X_val_gen_top.shape))
    else:
        X_tr_gen_top = X_val_gen_top = None

    # ── Build model ──────────────────────────────────────────────────────────────
    # Seed weight init so seed-sweep runs are reproducible.  calibration_seed is
    # kept separate so the calib/held-out split is not perturbed by training_seed.
    torch.manual_seed(cfg.training_seed)
    np.random.seed(cfg.training_seed)
    model = CrosscoderPairwiseReadout(
        n_features=crosscoder.num_latents,
        hidden_size=cfg.hidden_size,
        gate_hidden=cfg.gate_hidden_size,
        dropout=cfg.dropout,
        bilinear_k=bilinear_top_k_actual,
        bilinear_rank=cfg.bilinear_rank,
        triple_bilinear=cfg.use_triple_bilinear,
    ).to(device)
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    # ── Train (use calibration set as training-time val) ─────────────────────────
    best_epoch, best_calib_f1 = _train(
        model, X_tr_dense, y_tr_float,
        img_h_t, img_b_t, txt_h_t, txt_b_t,
        sample_weights=None, cfg=cfg, device=device,
        val_X=X_val_calib, val_y=y_val_calib,
        X_img_top=X_tr_img_top, X_pmt_top=X_tr_pmt_top, X_gen_top=X_tr_gen_top,
        val_X_img_top=X_val_img_top[calib_idx] if X_val_img_top is not None else None,
        val_X_pmt_top=X_val_pmt_top[calib_idx] if X_val_pmt_top is not None else None,
        val_X_gen_top=X_val_gen_top[calib_idx] if X_val_gen_top is not None else None,
    )
    logger.info("Training done: best_epoch=%d calib_f1=%.4f", best_epoch, best_calib_f1)

    # ── Inference on held-out val ────────────────────────────────────────────────
    def _infer(X: torch.Tensor, xi: torch.Tensor | None, xp: torch.Tensor | None,
               xg: torch.Tensor | None = None) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(
                X.to(device),
                xi.to(device) if xi is not None else None,
                xp.to(device) if xp is not None else None,
                xg.to(device) if xg is not None else None,
            )).cpu().numpy()

    held_scores  = _infer(X_val_held,  X_val_img_top[held_idx]  if X_val_img_top is not None else None,
                                        X_val_pmt_top[held_idx]  if X_val_pmt_top is not None else None,
                                        X_val_gen_top[held_idx]  if X_val_gen_top is not None else None)
    calib_scores = _infer(X_val_calib, X_val_img_top[calib_idx] if X_val_img_top is not None else None,
                                        X_val_pmt_top[calib_idx] if X_val_pmt_top is not None else None,
                                        X_val_gen_top[calib_idx] if X_val_gen_top is not None else None)
    all_scores   = _infer(X_val_dense, X_val_img_top if X_val_img_top is not None else None,
                                        X_val_pmt_top if X_val_pmt_top is not None else None,
                                        X_val_gen_top if X_val_gen_top is not None else None)

    # Threshold calibration on calib set only
    best_t = 0.5
    best_calib_metric = -1.0
    proba_calib = np.stack([1 - calib_scores, calib_scores], axis=1)
    best_t, _, best_cal_preds_calib = _calibrate_threshold(
        y_val_calib, proba_calib, ["not hateful", "hateful"], "hateful")
    calib_set_metrics = _per_class_metrics(y_val_calib, best_cal_preds_calib)

    # Apply threshold to held-out set
    held_preds_raw = ["hateful" if s >= 0.5 else "not hateful" for s in held_scores]
    held_preds_cal = ["hateful" if s >= best_t else "not hateful" for s in held_scores]
    raw_metrics = _per_class_metrics(y_val_held, held_preds_raw)
    cal_metrics  = _per_class_metrics(y_val_held, held_preds_cal)
    cal_metrics["threshold"] = round(best_t, 2)
    cal_metrics["calibration_note"] = "threshold tuned on calib split, reported on held-out split"

    logger.info("Held-out: raw_f1=%.4f cal_f1=%.4f (thr=%.2f)", raw_metrics["macro_f1"], cal_metrics["macro_f1"], best_t)

    # Baseline comparison (on held-out keys only)
    baseline_metrics = None
    baseline_by_key: dict[str, Any] = {}
    if baseline_path and baseline_path.exists():
        for row in _read_jsonl(baseline_path):
            key = f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"
            baseline_by_key[key] = row
        bg, bp = [], []
        for k, g in zip(val_keys_held, y_val_held):
            bpred = baseline_by_key.get(k, {}).get("prediction_label")
            if bpred in set(FHM_LABELS):
                bg.append(g); bp.append(bpred)
        if bg:
            baseline_metrics = _per_class_metrics(bg, bp)

    # Pair metrics on full val (pairs span both calib and held-out)
    all_preds = ["hateful" if s >= best_t else "not hateful" for s in all_scores]
    all_proba = [{"hateful": float(s), "not hateful": float(1 - s)} for s in all_scores]
    pred_by_key  = dict(zip(val_keys, all_preds))
    proba_by_key = dict(zip(val_keys, all_proba))
    pm = {src: _pair_metrics_local(val_pairs[src], pred_by_key, proba_by_key) for src in cfg.pair_sources}

    # Gate stats
    with torch.no_grad():
        gate_vals = model.gate(X_val_dense.to(device)).squeeze(-1).cpu().numpy()
    gate_stats = {"mean": round(float(gate_vals.mean()), 4), "std": round(float(gate_vals.std()), 4),
                  "min": round(float(gate_vals.min()), 4), "max": round(float(gate_vals.max()), 4)}
    logger.info("Gate: mean=%.4f std=%.4f", gate_stats["mean"], gate_stats["std"])

    # ── Save outputs ─────────────────────────────────────────────────────────────
    metrics = {
        "config": vars(cfg),
        "crosscoder_hookpoint": cfg.crosscoder_hookpoint,
        "n_features": crosscoder.num_latents,
        "n_train_real": len(train_keys),
        "n_train_total": int(X_train.shape[0]),
        "n_val_calib": len(calib_idx),
        "n_val_held": len(held_idx),
        "best_epoch": best_epoch,
        "calib_set_metrics": calib_set_metrics,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": cal_metrics,
        "baseline_metrics": baseline_metrics,
        "pair_metrics": {src: {k: v for k, v in m.items() if k != "rows"} for src, m in pm.items()},
        "gate_stats": gate_stats,
        "delta_stats": delta_stats,
    }
    (output_dir / "pairwise_probe_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Optional checkpoint: model weights + bilinear feature indices
    if cfg.save_checkpoint and cfg.use_bilinear and bilinear_top_k_actual > 0:
        torch.save(model.state_dict(), output_dir / "model_checkpoint.pt")
        (output_dir / "bilinear_feature_indices.json").write_text(
            json.dumps({"feature_indices": top_k_feats.tolist()}, indent=2)
        )
        logger.info("Saved checkpoint and bilinear feature indices to %s", output_dir)

    with (output_dir / "probe_val_predictions.jsonl").open("w") as fd:
        for key, gold, pred, proba in zip(val_keys, y_val_all, all_preds, all_proba):
            base_row = baseline_by_key.get(key)
            split = "calib" if key in {val_keys[i] for i in calib_idx} else "held"
            fd.write(json.dumps({
                "sample_key": key, "task_key": TASK_KEY,
                "gold_label": gold, "probe_pred": pred, "probe_proba": proba,
                "val_split": split,
                "baseline_pred": base_row.get("prediction_label") if base_row else None,
                "probe_correct": pred == gold,
                "baseline_correct": (base_row.get("prediction_label") == gold) if base_row else None,
            }) + "\n")
    with (output_dir / "validation_pair_metrics.jsonl").open("w") as fd:
        for src, m in pm.items():
            for row in m.get("rows", []):
                fd.write(json.dumps({"source": src, **row}) + "\n")

    logger.info("Wrote outputs to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(CrosscoderPairwiseConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run_crosscoder_pairwise(cfg)


if __name__ == "__main__":
    main()
