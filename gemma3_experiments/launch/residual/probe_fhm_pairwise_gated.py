

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.analyze_fhm_confounder_features import _build_pairs
from gemma3_experiments.launch.residual.probe_validate import (
    _calibrate_threshold,
    _extract_features,
    _extract_features_multi,
    _load_fhm_confounder_metadata,
    _normalise_fhm_id,
    _normalise_group_value,
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

TASK_KEY = "Hateful_Memes_Dataset::hateful_memes_binary::label"
FHM_CLASSES = ["hateful", "not hateful"]
_LABEL_TO_INT = {"hateful": 1, "not hateful": 0}


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

@dataclass
class PairwiseGatedConfig:
    # SAE / cache paths (same as probe_validate)
    residual_sae_path: str = ""
    train_cache_dir: str = ""
    val_cache_dir: str = ""
    extra_val_cache_dirs: list[str] = field(default_factory=list)
    baseline_predictions_path: str = ""
    fhm_confounder_metadata_path: str = ""
    output_dir: str = ""

    # Feature extraction — use all tokens, same as best FHM probe
    token_selection: str = "all"
    aggregation: str = "max"
    batch_size: int = 256

    # IVL pair sources
    pair_sources: list[str] = field(default_factory=lambda: ["pseudo_img", "pseudo_text"])

    # Model architecture
    hidden_size: int = 256      # per-head hidden dim (shared encoder output)
    gate_hidden_size: int = 32  # gate MLP hidden dim
    dropout: float = 0.2

    # Training
    n_epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-3
    patience: int = 10          # patience in units of eval_every epochs
    eval_every: int = 10        # evaluate val F1 every N epochs

    # Combined loss
    margin: float = 0.2         # hinge margin for pair loss
    alpha: float = 0.85         # CE weight; (1-alpha-beta) = pair margin weight
    # BCE positive class weight: amplifies gradient for hateful samples.
    # Values > 1 push the model to recall more hateful examples.
    # Required when alpha > 0.7 to prevent collapse to all-not-hateful;
    # the pair hinge provides an upward gradient on hateful scores that
    # becomes insufficient when pair_weight < ~0.25.
    pos_weight: float = 1.0
    # Add pseudo-HN delta rows to training data (feature_h - feature_b → hateful
    # and feature_b - feature_h → not hateful).  Mirrors MLP+pseudo-HN approach
    # from probe_validate.py; improves AUROC by teaching sharper pair separation
    # in feature space, complementing the pairwise hinge loss at the output.
    add_delta_rows: bool = True
    # Explicitly supervise the gate for samples exclusively in one pair source:
    #   img-only samples → gate target = 1 (route to img_head)
    #   text-only samples → gate target = 0 (route to text_head)
    # Weight beta; 0 to disable.  Applied after alpha_warmup_epochs.
    gate_supervision_weight: float = 0.1
    # Warm up with CE-only for first N epochs before enabling pair + gate losses.
    # Prevents the pair hinge from destabilising early training.
    alpha_warmup_epochs: int = 30

    # Calibration
    calibrate_threshold: bool = True

    # Eval-only mode: skip training, load saved model, run inference on eval cache.
    # Use to get test_seen numbers without retraining.
    eval_only: bool = False
    eval_model_path: str = ""   # path to model.pt saved by a previous run
    eval_splits: list[str] = field(default_factory=lambda: ["test", "test_seen"])

    device: str = "cuda:0"
    seed: int = 42


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

class GatedPairwiseReadout(nn.Module):
    pass

    def __init__(
        self,
        n_features: int,
        gate_dim: int = 3,
        hidden: int = 256,
        gate_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Separate classification heads per confounder type
        self.img_head = nn.Linear(hidden, 1)    # pseudo_img: visual disambiguation
        self.text_head = nn.Linear(hidden, 1)   # pseudo_text: semantic disambiguation
        # Gate: non-label metadata → P(route to img_head)
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Return (B,) unnormalised logit scores (positive = hateful)."""
        h = self.encoder(x)
        s_img = self.img_head(h).squeeze(-1)    # (B,)
        s_txt = self.text_head(h).squeeze(-1)   # (B,)
        g = self.gate(meta).squeeze(-1)          # (B,) in [0, 1]
        return g * s_img + (1.0 - g) * s_txt


# ──────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────

def _fast_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via Wilcoxon-Mann-Whitney rank statistic.  O(n log n), no sklearn."""
    n = len(scores)
    order = np.argsort(scores)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pair_hinge_loss(
    scores: torch.Tensor,
    pairs: list[tuple[str, str]],
    key_to_idx: dict[str, int],
    margin: float,
) -> torch.Tensor:
    """Mean hinge: max(0, score(benign) - score(hateful) + margin) over all pairs."""
    terms: list[torch.Tensor] = []
    for h_key, b_key in pairs:
        h_i = key_to_idx.get(h_key)
        b_i = key_to_idx.get(b_key)
        if h_i is None or b_i is None:
            continue
        terms.append(torch.clamp(scores[b_i] - scores[h_i] + margin, min=0.0))
    if not terms:
        return scores.new_zeros(())
    return torch.stack(terms).mean()


# ──────────────────────────────────────────────────────────────
# Gate feature construction
# ──────────────────────────────────────────────────────────────

def _build_gate_features(
    keys: list[str],
    csr: Any,
    metadata: dict[str, dict[str, Any]],
    pairs_by_source: dict[str, list[tuple[str, str]]],
) -> np.ndarray:
    """Build (N, 3) gate feature matrix (float32):
        [0] log1p(L2 norm of feature vector)       -- proxy for content richness
        [1] log1p(OCR text length from IVL meta)   -- proxy for text complexity
        [2] pair_source_ratio in [-1, 1]            -- (n_img - n_text) / (n_total + 1)
                                                       0 for samples not in any pair
    """
    key_to_idx = {k: i for i, k in enumerate(keys)}

    # Feature L2 norms
    norms = np.sqrt(np.array(csr.power(2).sum(axis=1)).ravel())

    # OCR text length from IVL metadata
    text_lens = np.zeros(len(keys), dtype=np.float32)
    for i, key in enumerate(keys):
        sample_id = key.split("::")[-1]
        meta = metadata.get(_normalise_fhm_id(sample_id)) or {}
        text = str(meta.get("text") or meta.get("meme_text") or "")
        text_lens[i] = float(len(text))

    # Pair source ratio: how much is each sample in pseudo_img vs pseudo_text pairs?
    n_img = np.zeros(len(keys), dtype=np.float32)
    n_txt = np.zeros(len(keys), dtype=np.float32)
    for h_key, b_key in pairs_by_source.get("pseudo_img", []):
        for k in (h_key, b_key):
            idx = key_to_idx.get(k)
            if idx is not None:
                n_img[idx] += 1.0
    for h_key, b_key in pairs_by_source.get("pseudo_text", []):
        for k in (h_key, b_key):
            idx = key_to_idx.get(k)
            if idx is not None:
                n_txt[idx] += 1.0
    ratio = (n_img - n_txt) / (n_img + n_txt + 1.0)

    return np.stack([
        np.log1p(norms).astype(np.float32),
        np.log1p(text_lens),
        ratio,
    ], axis=1)


# ──────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────

def _pair_margin_stats(
    proba_hat: np.ndarray,
    pairs: list[tuple[str, str]],
    key_to_idx: dict[str, int],
) -> dict[str, Any]:
    """P(hateful) margin stats for pseudo-confounder pairs."""
    margins: list[float] = []
    for h_key, b_key in pairs:
        h_i = key_to_idx.get(h_key)
        b_i = key_to_idx.get(b_key)
        if h_i is None or b_i is None:
            continue
        margins.append(float(proba_hat[h_i]) - float(proba_hat[b_i]))
    if not margins:
        return {"n_pairs": 0}
    m = np.array(margins)
    return {
        "n_pairs": len(m),
        "mean_margin": round(float(np.mean(m)), 6),
        "median_margin": round(float(np.median(m)), 6),
        "positive_margin_rate": round(float(np.mean(m > 0)), 4),
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def _run_eval_only(cfg: "PairwiseGatedConfig") -> None:
    """Load a saved model.pt and evaluate on the test (or any) split.

    Mirrors the evaluation section of run() but skips all training.
    Use this to get test_seen numbers without retraining.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert cfg.eval_model_path, "eval_only=True requires eval_model_path"
    model_path = Path(cfg.eval_model_path)
    assert model_path.exists(), f"Model checkpoint not found: {model_path}"

    device = resolve_visible_device(cfg.device)
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)
    n_features = sae.num_latents

    metadata = _load_fhm_confounder_metadata(Path(cfg.fhm_confounder_metadata_path))

    extract_kw = dict(
        token_selection=cfg.token_selection,
        aggregation=cfg.aggregation,
        batch_size=cfg.batch_size,
        device=device,
        n_features=n_features,
    )

    # Load eval rows from val_cache_dir (point this at residual_test for test_seen)
    eval_cache = Path(cfg.val_cache_dir)
    all_eval_caches = [eval_cache] + [
        Path(d) for d in cfg.extra_val_cache_dirs if Path(d).exists()
    ]
    eval_rows_all: list[dict[str, Any]] = []
    for vc in all_eval_caches:
        for r in _read_jsonl(vc / "manifest.jsonl"):
            r["_cache_dir"] = str(vc)
            eval_rows_all.append(r)
    eval_rows = [
        r for r in eval_rows_all
        if r.get("dataset_name") == "Hateful_Memes_Dataset"
        and r.get("split") in set(cfg.eval_splits)
        and _row_gold_label(r) in set(FHM_CLASSES)
    ]
    logger.info("Eval rows (splits=%s): %d", cfg.eval_splits, len(eval_rows))
    assert eval_rows, f"No labeled FHM rows found for splits {cfg.eval_splits}"

    eval_keys, eval_csr = _extract_features_multi(eval_rows, sae, stats, **extract_kw)
    eval_rows_by_key = {_row_sample_key(r): r for r in eval_rows}
    eval_labels = [_row_gold_label(eval_rows_by_key[k]) for k in eval_keys]
    eval_key_to_idx = {k: i for i, k in enumerate(eval_keys)}
    logger.info("Eval CSR: %d x %d", *eval_csr.shape)

    # Build pairs and gate features
    eval_key_set = set(eval_keys)
    eval_pairs = {s: _build_pairs(eval_rows, metadata, eval_key_set, s) for s in cfg.pair_sources}
    eval_gate = _build_gate_features(eval_keys, eval_csr, metadata, eval_pairs)
    gate_dim = eval_gate.shape[1]

    # Load model
    ckpt = torch.load(str(model_path), map_location="cpu")
    model = GatedPairwiseReadout(
        n_features=ckpt.get("n_features", n_features),
        gate_dim=ckpt.get("gate_dim", gate_dim),
        hidden=ckpt.get("cfg", {}).get("hidden_size", cfg.hidden_size),
        gate_hidden=ckpt.get("cfg", {}).get("gate_hidden_size", cfg.gate_hidden_size),
        dropout=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded model from %s", model_path)

    X_ev = torch.from_numpy(eval_csr.toarray()).float().to(device)
    M_ev = torch.from_numpy(eval_gate).float().to(device)
    y_ev_bin = np.array([_LABEL_TO_INT[l] for l in eval_labels], dtype=np.float32)

    with torch.inference_mode():
        scores = model(X_ev, M_ev)
        proba_hat = torch.sigmoid(scores).cpu().numpy()

    pred = ["hateful" if p >= 0.5 else "not hateful" for p in proba_hat]
    raw_metrics = _per_class_metrics(eval_labels, pred)

    from sklearn.metrics import roc_auc_score
    try:
        auroc = float(roc_auc_score(y_ev_bin, proba_hat))
    except ValueError:
        auroc = float("nan")

    calibrated_metrics = None
    if cfg.calibrate_threshold:
        proba_2d = np.stack([1.0 - proba_hat, proba_hat], axis=1)
        best_t, _, best_preds = _calibrate_threshold(
            eval_labels, proba_2d, ["not hateful", "hateful"], "hateful"
        )
        if best_preds:
            calibrated_metrics = _per_class_metrics(eval_labels, best_preds)
            calibrated_metrics["threshold"] = round(best_t, 2)

    pair_metrics = {
        src: _pair_margin_stats(proba_hat, pairs, eval_key_to_idx)
        for src, pairs in eval_pairs.items()
    }

    metrics = {
        "eval_splits": cfg.eval_splits,
        "eval_model_path": str(model_path),
        "n_eval": len(eval_keys),
        "n_eval_pairs": {s: len(p) for s, p in eval_pairs.items()},
        "auroc": round(auroc, 6),
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "pair_metrics": pair_metrics,
    }
    suffix = "_".join(cfg.eval_splits)
    out_path = output_dir / f"pairwise_gated_metrics_{suffix}.json"
    out_path.write_text(json.dumps(metrics, indent=2))

    pred_path = output_dir / f"probe_val_predictions_{suffix}.jsonl"
    with pred_path.open("w") as fd:
        for key, gold, p, p_hat in zip(eval_keys, eval_labels, pred, proba_hat):
            fd.write(json.dumps({
                "sample_key": key,
                "task_key": TASK_KEY,
                "gold_label": gold,
                "probe_pred": p,
                "probe_proba": {"hateful": round(float(p_hat), 6), "not hateful": round(float(1.0 - p_hat), 6)},
                "probe_correct": p == gold,
            }) + "\n")

    logger.info(
        "Eval (%s): n=%d  raw_f1=%.4f  AUROC=%.4f  hateful_recall=%.4f  cal_f1=%.4f (t=%.2f)",
        suffix, len(eval_keys), raw_metrics["macro_f1"], auroc,
        raw_metrics["per_class"].get("hateful", {}).get("recall", 0.0),
        calibrated_metrics["macro_f1"] if calibrated_metrics else float("nan"),
        calibrated_metrics.get("threshold", 0.5) if calibrated_metrics else 0.5,
    )
    logger.info("Wrote %s and %s", out_path, pred_path)


def run(cfg: PairwiseGatedConfig) -> None:
    if cfg.eval_only:
        _run_eval_only(cfg)
        return

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_visible_device(cfg.device)
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)
    n_features = sae.num_latents
    logger.info("SAE num_latents=%d", n_features)

    metadata = _load_fhm_confounder_metadata(Path(cfg.fhm_confounder_metadata_path))
    logger.info("IVL metadata: %d entries", len(metadata))

    extract_kw = dict(
        token_selection=cfg.token_selection,
        aggregation=cfg.aggregation,
        batch_size=cfg.batch_size,
        device=device,
        n_features=n_features,
    )

    # ── Load FHM train rows ───────────────────────────────────────────────────
    train_cache = Path(cfg.train_cache_dir)
    train_rows_all = _read_jsonl(train_cache / "manifest.jsonl")
    train_rows = [
        r for r in train_rows_all
        if r.get("dataset_name") == "Hateful_Memes_Dataset"
        and r.get("split") == "train"
        and _row_gold_label(r) in set(FHM_CLASSES)
    ]
    logger.info("FHM train rows: %d", len(train_rows))
    assert train_rows, "No FHM train rows found"

    logger.info("Extracting train features (token_selection=%s)...", cfg.token_selection)
    train_keys, train_csr = _extract_features(train_rows, train_cache, sae, stats, **extract_kw)
    train_rows_by_key = {_row_sample_key(r): r for r in train_rows}
    logger.info("Train CSR: %d x %d  nnz=%d", *train_csr.shape, train_csr.nnz)

    # ── Load FHM val rows ─────────────────────────────────────────────────────
    val_cache = Path(cfg.val_cache_dir)
    all_val_caches = [val_cache] + [
        Path(d) for d in cfg.extra_val_cache_dirs if Path(d).exists()
    ]
    val_rows_all: list[dict[str, Any]] = []
    for vc in all_val_caches:
        for r in _read_jsonl(vc / "manifest.jsonl"):
            r["_cache_dir"] = str(vc)
            val_rows_all.append(r)
    val_rows = [
        r for r in val_rows_all
        if r.get("dataset_name") == "Hateful_Memes_Dataset"
        and r.get("split") in {"validation", "val", "dev"}
        and _row_gold_label(r) in set(FHM_CLASSES)
    ]
    logger.info("FHM val rows: %d", len(val_rows))
    assert val_rows, "No FHM val rows found"

    logger.info("Extracting val features...")
    val_keys, val_csr = _extract_features_multi(val_rows, sae, stats, **extract_kw)
    val_rows_by_key = {_row_sample_key(r): r for r in val_rows}
    logger.info("Val CSR: %d x %d  nnz=%d", *val_csr.shape, val_csr.nnz)

    # ── Build pseudo-confounder pairs ─────────────────────────────────────────
    train_key_set = set(train_keys)
    val_key_set = set(val_keys)
    train_pairs = {s: _build_pairs(train_rows, metadata, train_key_set, s) for s in cfg.pair_sources}
    val_pairs = {s: _build_pairs(val_rows, metadata, val_key_set, s) for s in cfg.pair_sources}
    for src, pairs in train_pairs.items():
        logger.info("Train pairs %-12s: %d", src, len(pairs))
    for src, pairs in val_pairs.items():
        logger.info("Val pairs   %-12s: %d", src, len(pairs))

    all_train_pairs: list[tuple[str, str]] = []
    for src_pairs in train_pairs.values():
        all_train_pairs.extend(src_pairs)
    logger.info("Total train pairs for pair loss: %d", len(all_train_pairs))

    # ── Gate features ─────────────────────────────────────────────────────────
    train_gate = _build_gate_features(train_keys, train_csr, metadata, train_pairs)
    val_gate = _build_gate_features(val_keys, val_csr, metadata, val_pairs)
    gate_dim = train_gate.shape[1]
    logger.info("Gate features (dim=%d): train=%s val=%s", gate_dim, train_gate.shape, val_gate.shape)

    # Log gate feature statistics for interpretability
    for i, name in enumerate(["log_l2_norm", "log_text_len", "pair_source_ratio"]):
        logger.info("  gate[%d] %-20s train: mean=%.3f std=%.3f", i, name,
                    train_gate[:, i].mean(), train_gate[:, i].std())

    # ── Labels ────────────────────────────────────────────────────────────────
    train_labels = [_row_gold_label(train_rows_by_key[k]) for k in train_keys]
    val_labels = [_row_gold_label(val_rows_by_key[k]) for k in val_keys]
    y_train = np.array([_LABEL_TO_INT[l] for l in train_labels], dtype=np.float32)

    # ── Convert to dense tensors (650 MB for FHM all-token train) ─────────────
    logger.info("Converting CSR to dense tensors...")
    X_train = torch.from_numpy(train_csr.toarray()).float()
    X_val = torch.from_numpy(val_csr.toarray()).float()
    M_train = torch.from_numpy(train_gate).float()
    M_val = torch.from_numpy(val_gate).float()
    Y_train = torch.from_numpy(y_train)

    train_key_to_idx = {k: i for i, k in enumerate(train_keys)}
    val_key_to_idx = {k: i for i, k in enumerate(val_keys)}

    # ── Gate supervision targets ──────────────────────────────────────────────
    # Supervise the gate for samples that are exclusively in one pair source:
    #   n_img > 0 and n_txt == 0  → target 1.0 (route to img_head)
    #   n_img == 0 and n_txt > 0  → target 0.0 (route to text_head)
    # Ambiguous (in both or neither) are excluded from gate supervision.
    n_img_counts = np.zeros(len(train_keys), dtype=np.float32)
    n_txt_counts = np.zeros(len(train_keys), dtype=np.float32)
    for h_key, b_key in train_pairs.get("pseudo_img", []):
        for k in (h_key, b_key):
            idx = train_key_to_idx.get(k)
            if idx is not None:
                n_img_counts[idx] += 1.0
    for h_key, b_key in train_pairs.get("pseudo_text", []):
        for k in (h_key, b_key):
            idx = train_key_to_idx.get(k)
            if idx is not None:
                n_txt_counts[idx] += 1.0
    img_only_mask = (n_img_counts > 0) & (n_txt_counts == 0)
    txt_only_mask = (n_img_counts == 0) & (n_txt_counts > 0)
    gate_sup_mask = img_only_mask | txt_only_mask
    gate_sup_targets_np = np.where(img_only_mask, 1.0, 0.0).astype(np.float32)
    gate_sup_idx = torch.from_numpy(np.where(gate_sup_mask)[0]).long()
    gate_sup_targets = torch.from_numpy(gate_sup_targets_np[gate_sup_mask]).float()
    logger.info(
        "Gate supervision: %d samples total  (img-only=%d  text-only=%d  ambiguous=%d)",
        int(gate_sup_mask.sum()), int(img_only_mask.sum()),
        int(txt_only_mask.sum()), int(((n_img_counts > 0) & (n_txt_counts > 0)).sum()),
    )
    gate_sup_idx_d = gate_sup_idx.to(device)
    gate_sup_targets_d = gate_sup_targets.to(device)
    gate_sup_fn = nn.BCELoss()

    if cfg.add_delta_rows and all_train_pairs:
        from scipy.sparse import vstack as _sp_vstack
        _delta_rows, _delta_y = [], []
        _seen_pairs: set[tuple[str, str]] = set()
        for h_key, b_key in all_train_pairs:
            pk = tuple(sorted((h_key, b_key)))
            if pk in _seen_pairs:
                continue
            _seen_pairs.add(pk)
            h_i = train_key_to_idx.get(h_key)
            b_i = train_key_to_idx.get(b_key)
            if h_i is None or b_i is None:
                continue
            _delta_rows.append(train_csr[h_i] - train_csr[b_i])
            _delta_y.append(1.0)
            _delta_rows.append(train_csr[b_i] - train_csr[h_i])
            _delta_y.append(0.0)
        if _delta_rows:
            _delta_csr = _sp_vstack(_delta_rows, format="csr", dtype=np.float32)
            _aug_csr = _sp_vstack([train_csr, _delta_csr], format="csr", dtype=np.float32)
            _aug_gate = np.vstack([train_gate, np.zeros((len(_delta_y), gate_dim), np.float32)])
            _aug_y = np.concatenate([y_train, np.array(_delta_y, np.float32)])
            logger.info(
                "Delta rows: %d unique pairs → %d synthetic rows; train %d → %d",
                len(_seen_pairs), len(_delta_y), len(train_keys), len(train_keys) + len(_delta_y),
            )
        else:
            _aug_csr, _aug_gate, _aug_y = train_csr, train_gate, y_train
    else:
        _aug_csr, _aug_gate, _aug_y = train_csr, train_gate, y_train

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model = GatedPairwiseReadout(
        n_features=n_features,
        gate_dim=gate_dim,
        hidden=cfg.hidden_size,
        gate_hidden=cfg.gate_hidden_size,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d  (encoder: %d  heads: %d  gate: %d)",
                n_params,
                sum(p.numel() for p in model.encoder.parameters()),
                sum(p.numel() for p in model.img_head.parameters()) +
                sum(p.numel() for p in model.text_head.parameters()),
                sum(p.numel() for p in model.gate.parameters()))

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.n_epochs, eta_min=cfg.lr * 0.01
    )
    ce_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg.pos_weight], dtype=torch.float32).to(device)
        if cfg.pos_weight != 1.0 else None
    )

    # Move everything to device once (full-batch training)
    X_tr = torch.from_numpy(_aug_csr.toarray()).float().to(device)
    M_tr = torch.from_numpy(_aug_gate).float().to(device)
    Y_tr = torch.from_numpy(_aug_y).to(device)
    X_va = X_val.to(device)
    M_va = M_val.to(device)
    y_val_bin = np.array([_LABEL_TO_INT[l] for l in val_labels], dtype=np.float32)

    # ── Training loop (full-batch) ─────────────────────────────────────────────
    best_val_f1 = 0.0
    best_state: dict | None = None
    best_val_auroc = 0.0
    best_auroc_state: dict | None = None
    patience_counter = 0

    beta = cfg.gate_supervision_weight
    assert cfg.alpha + beta <= 1.0, f"alpha ({cfg.alpha}) + beta ({beta}) must be <= 1"
    pair_weight = 1.0 - cfg.alpha - beta
    logger.info(
        "Training: n_epochs=%d  alpha=%.2f (CE)  pair_weight=%.2f  beta=%.2f (gate_sup)  "
        "margin=%.2f  warmup=%d  lr=%.0e",
        cfg.n_epochs, cfg.alpha, pair_weight, beta, cfg.margin, cfg.alpha_warmup_epochs, cfg.lr,
    )

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        optimiser.zero_grad()

        scores_tr = model(X_tr, M_tr)              # (N_train,)
        loss_ce = ce_fn(scores_tr, Y_tr)

        # Enable pair + gate supervision losses after warmup
        warming_up = epoch <= cfg.alpha_warmup_epochs
        if warming_up:
            loss = loss_ce
            loss_pair = scores_tr.new_zeros(())
            loss_gate = scores_tr.new_zeros(())
        else:
            loss_pair = _pair_hinge_loss(scores_tr, all_train_pairs, train_key_to_idx, cfg.margin)
            # Gate supervision: push gate outputs toward correct routing for unambiguous samples
            gate_out = model.gate(M_tr).squeeze(-1)  # (N_train,)
            loss_gate = gate_sup_fn(
                gate_out[gate_sup_idx_d], gate_sup_targets_d
            ) if gate_sup_idx_d.numel() > 0 else scores_tr.new_zeros(())
            loss = cfg.alpha * loss_ce + pair_weight * loss_pair + beta * loss_gate

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        scheduler.step()

        if epoch % cfg.eval_every == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                val_scores = model(X_va, M_va)
                val_proba = torch.sigmoid(val_scores).cpu().numpy()
                gate_weights = model.gate(M_tr).squeeze(-1).cpu().numpy()

            val_pred = ["hateful" if p >= 0.5 else "not hateful" for p in val_proba]
            vm = _per_class_metrics(val_labels, val_pred)
            hateful_r = vm["per_class"].get("hateful", {}).get("recall", 0.0)
            val_auroc = _fast_auroc(val_proba, y_val_bin)

            logger.info(
                "Epoch %3d%s  loss=%.4f ce=%.4f pair=%.4f gate_sup=%.4f | "
                "val_f1=%.4f AUROC=%.4f hateful_r=%.3f | "
                "gate: mean=%.3f std=%.3f",
                epoch, " (warm)" if warming_up else "       ",
                loss.item(), loss_ce.item(), loss_pair.item(), loss_gate.item(),
                vm["macro_f1"], val_auroc, hateful_r,
                gate_weights.mean(), gate_weights.std(),
            )

            if vm["macro_f1"] > best_val_f1:
                best_val_f1 = vm["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_auroc_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Final evaluation: best-F1 checkpoint ───────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.inference_mode():
        val_scores = model(X_va, M_va)
        val_proba_hat = torch.sigmoid(val_scores).cpu().numpy()   # (N_val,) P(hateful)
        gate_val = model.gate(M_va).squeeze(-1).cpu().numpy()

    val_pred = ["hateful" if p >= 0.5 else "not hateful" for p in val_proba_hat]
    raw_metrics = _per_class_metrics(val_labels, val_pred)

    # AUROC — threshold-independent; primary FHM benchmark metric
    from sklearn.metrics import roc_auc_score
    y_val_binary = np.array([_LABEL_TO_INT[l] for l in val_labels], dtype=np.float32)
    try:
        auroc = float(roc_auc_score(y_val_binary, val_proba_hat))
    except ValueError:
        auroc = float("nan")

    # ── Also evaluate best-AUROC checkpoint (may differ from best-F1) ────────────
    auroc_ckpt_metrics: dict | None = None
    if best_auroc_state is not None and best_auroc_state is not best_state:
        model.load_state_dict(best_auroc_state)
        model.eval()
        with torch.inference_mode():
            auc_scores = model(X_va, M_va)
            auc_proba = torch.sigmoid(auc_scores).cpu().numpy()
        auc_pred = ["hateful" if p >= 0.5 else "not hateful" for p in auc_proba]
        auc_metrics = _per_class_metrics(val_labels, auc_pred)
        try:
            auc_auroc = float(roc_auc_score(y_val_binary, auc_proba))
        except ValueError:
            auc_auroc = float("nan")
        auroc_ckpt_metrics = {"macro_f1": auc_metrics["macro_f1"], "auroc": round(auc_auroc, 6)}
        logger.info("Best-AUROC ckpt: macro_f1=%.4f AUROC=%.4f  (best-F1 ckpt: macro_f1=%.4f AUROC=%.4f)",
                    auc_metrics["macro_f1"], auc_auroc, raw_metrics["macro_f1"], auroc)
        model.load_state_dict(best_state)  # restore best-F1 for saving

    logger.info("Raw: accuracy=%.4f macro_f1=%.4f hateful_recall=%.4f AUROC=%.4f",
                raw_metrics["accuracy"], raw_metrics["macro_f1"],
                raw_metrics["per_class"].get("hateful", {}).get("recall", 0.0),
                auroc)

    # Gate analysis: are img/text pair members routed differently?
    gate_by_source: dict[str, dict[str, float]] = {}
    for src, pairs in val_pairs.items():
        gate_for_src = []
        for h_key, b_key in pairs:
            for k in (h_key, b_key):
                idx = val_key_to_idx.get(k)
                if idx is not None:
                    gate_for_src.append(float(gate_val[idx]))
        if gate_for_src:
            gate_by_source[src] = {
                "mean": round(float(np.mean(gate_for_src)), 4),
                "std": round(float(np.std(gate_for_src)), 4),
            }
    logger.info("Gate weights by pair source: %s", gate_by_source)

    # Threshold calibration
    calibrated_metrics = None
    if cfg.calibrate_threshold:
        proba_2d = np.stack([1.0 - val_proba_hat, val_proba_hat], axis=1)
        best_t, _, best_preds = _calibrate_threshold(
            val_labels, proba_2d, ["not hateful", "hateful"], "hateful"
        )
        if best_preds:
            calibrated_metrics = _per_class_metrics(val_labels, best_preds)
            calibrated_metrics["threshold"] = round(best_t, 2)
            logger.info("Calibrated (t=%.2f): accuracy=%.4f macro_f1=%.4f",
                        best_t, calibrated_metrics["accuracy"], calibrated_metrics["macro_f1"])

    # Pair margin stats
    pair_metrics = {
        src: _pair_margin_stats(val_proba_hat, pairs, val_key_to_idx)
        for src, pairs in val_pairs.items()
    }
    for src, pm in pair_metrics.items():
        logger.info("Pair %-12s: n=%d mean_margin=%.4f positive_rate=%.4f",
                    src, pm.get("n_pairs", 0), pm.get("mean_margin", 0), pm.get("positive_margin_rate", 0))

    # Baseline comparison
    baseline_pred: dict[str, str] = {}
    if cfg.baseline_predictions_path:
        for r in _read_jsonl(Path(cfg.baseline_predictions_path)):
            key = f"{r['dataset_name']}::{r['task_name']}::{r['sample_id']}"
            baseline_pred[key] = r.get("prediction_label", "")
    bl_gold = [g for k, g in zip(val_keys, val_labels) if baseline_pred.get(k) in set(FHM_CLASSES)]
    bl_pred_list = [baseline_pred[k] for k in val_keys if baseline_pred.get(k) in set(FHM_CLASSES)]
    baseline_metrics = _per_class_metrics(bl_gold, bl_pred_list) if bl_gold else None

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "task_key": TASK_KEY,
        "config": {
            "token_selection": cfg.token_selection,
            "hidden_size": cfg.hidden_size,
            "n_epochs_trained": sum(1 for _ in range(cfg.n_epochs)),
            "alpha": cfg.alpha,
            "margin": cfg.margin,
            "lr": cfg.lr,
        },
        "n_train": len(train_keys),
        "n_val": len(val_keys),
        "n_train_pairs": {s: len(p) for s, p in train_pairs.items()},
        "n_val_pairs": {s: len(p) for s, p in val_pairs.items()},
        "best_val_f1_during_training": round(best_val_f1, 6),
        "best_val_auroc_during_training": round(best_val_auroc, 6),
        "auroc_ckpt_vs_f1_ckpt": auroc_ckpt_metrics,
        "auroc": round(auroc, 6),
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
        "baseline_metrics": baseline_metrics,
        "pair_metrics": pair_metrics,
        "gate_by_source": gate_by_source,
        "macro_f1_delta_vs_baseline": round(
            raw_metrics["macro_f1"] - baseline_metrics["macro_f1"], 4
        ) if baseline_metrics else None,
    }
    (output_dir / "pairwise_gated_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Save predictions (probe_proba format, ablation-compatible) ────────────
    with (output_dir / "probe_val_predictions.jsonl").open("w") as fd:
        for key, gold, pred, p_hat in zip(val_keys, val_labels, val_pred, val_proba_hat):
            bp = baseline_pred.get(key)
            fd.write(json.dumps({
                "sample_key": key,
                "task_key": TASK_KEY,
                "gold_label": gold,
                "probe_pred": pred,
                "probe_proba": {
                    "hateful": round(float(p_hat), 6),
                    "not hateful": round(float(1.0 - p_hat), 6),
                },
                "baseline_pred": bp,
                "probe_correct": pred == gold,
                "baseline_correct": (bp == gold) if bp else None,
            }) + "\n")

    # ── Save model checkpoint ─────────────────────────────────────────────────
    torch.save({
        "model_state_dict": best_state or model.state_dict(),
        "n_features": n_features,
        "gate_dim": gate_dim,
        "cfg": cfg.__dict__,
    }, output_dir / "model.pt")

    logger.info(
        "Pairwise gated readout: raw_f1=%.4f  calibrated_f1=%.4f  AUROC=%.4f  baseline_f1=%.4f",
        raw_metrics["macro_f1"],
        calibrated_metrics["macro_f1"] if calibrated_metrics else float("nan"),
        auroc,
        baseline_metrics["macro_f1"] if baseline_metrics else float("nan"),
    )
    logger.info("Wrote %s", output_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(PairwiseGatedConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
