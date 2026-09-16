

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file
from simple_parsing import parse, Serializable, list_field

from gemma3_experiments.residual_sae.normalization import load_residual_stats
from ..models import load_residual_sae

logger = logging.getLogger(__name__)

# Token role constants (match cache_dense.py)
ROLE_PROMPT = 0
ROLE_IMAGE = 1
ROLE_GENERATED = 2

FHM_DATASET = "Hateful_Memes_Dataset"
FHM_TASK = "hateful_memes_binary"
FHM_POS_LABEL = "hateful"


@dataclass
class FHMBilinearConfig(Serializable):
    train_cache_dir: str = "./qwen_residual_sae_cache/residual"
    val_cache_dir: str = "./qwen_residual_sae_cache/residual_val_test"
    residual_sae_path: str = "./qwen_residual_sae_ckpts/best_finite"
    output_dir: str = "./qwen_fhm_bilinear"
    device: str = "cuda:0"

    # Feature selection
    top_k_features: int = 256        # number of features to select per role

    # Bilinear model
    bilinear_rank: int = 16          # rank of the bilinear interaction term
    n_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-3
    patience: int = 20               # early stopping patience (epochs without val improvement)

    # Which val/test splits to evaluate on
    eval_splits: list[str] = list_field("validation", "val", "dev", "test", "test_seen")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fd:
        for line in fd:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _role_mask(roles: torch.Tensor, role: int) -> torch.Tensor:
    return roles == role


class BilinearProbe(nn.Module):
    """Low-rank bilinear probe: score = bil + img_linear + pmt_linear + bias."""

    def __init__(self, k_img: int, k_pmt: int, rank: int):
        super().__init__()
        # Low-rank bilinear interaction: W_bil ≈ U @ V.T
        self.U = nn.Parameter(torch.empty(k_img, rank))
        self.V = nn.Parameter(torch.empty(k_pmt, rank))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        self.w_img = nn.Parameter(torch.zeros(k_img))
        self.w_pmt = nn.Parameter(torch.zeros(k_pmt))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_img: torch.Tensor, x_pmt: torch.Tensor) -> torch.Tensor:
        # x_img: [N, k_img], x_pmt: [N, k_pmt]
        bil = ((x_img @ self.U) * (x_pmt @ self.V)).sum(dim=-1, keepdim=True)  # [N,1]
        img_lin = (x_img * self.w_img).sum(dim=-1, keepdim=True)                # [N,1]
        pmt_lin = (x_pmt * self.w_pmt).sum(dim=-1, keepdim=True)               # [N,1]
        return (bil + img_lin + pmt_lin + self.bias).squeeze(-1)                # [N]


def _extract_role_features(
    sae,
    residual_stats,
    cache_dir: Path,
    manifest_rows: list[dict],
    selected_img_feats: np.ndarray | None,
    selected_pmt_feats: np.ndarray | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract image-role and prompt-role max-pooled SAE features.

    Returns (X_img, X_pmt, y) arrays where X_img/X_pmt are restricted to
    selected_img_feats/selected_pmt_feats if provided (else full d_sae).
    """
    by_shard: dict[str, list] = defaultdict(list)
    for row in manifest_rows:
        by_shard[row["shard_file"]].append(row)

    X_img_list: list[np.ndarray] = []
    X_pmt_list: list[np.ndarray] = []
    y_list: list[int] = []

    for shard_name, rows in sorted(by_shard.items()):
        shard_path = cache_dir / shard_name
        if not shard_path.exists():
            logger.warning("Missing shard: %s", shard_path)
            continue
        try:
            tensors = load_file(str(shard_path))
        except Exception as e:
            logger.warning("Failed to load %s: %s", shard_path, e)
            continue

        residual_all = tensors["residual"].to(device=device, dtype=torch.float32)
        roles_all = tensors.get("token_roles")

        for row in rows:
            offset, length = row["offset"], row["length"]
            sample_res = residual_all[offset: offset + length]
            if sample_res.shape[0] == 0:
                continue

            # Gold label
            gold = row.get("gold_fields", {})
            if not gold:
                continue
            gold_val = str(list(gold.values())[0])
            label = 1 if gold_val.lower() == FHM_POS_LABEL.lower() else 0

            # SAE encoding
            sae_in = residual_stats.normalize(sample_res) if residual_stats else sample_res
            with torch.no_grad():
                acts, _ = sae.encode(sae_in)  # [L, d_sae]

            if roles_all is not None:
                roles = roles_all[offset: offset + length]
                img_mask = _role_mask(roles, ROLE_IMAGE)
                pmt_mask = _role_mask(roles, ROLE_PROMPT)
            else:
                # Fallback: split half-half (not ideal)
                mid = length // 2
                img_mask = torch.zeros(length, dtype=torch.bool)
                img_mask[:mid] = True
                pmt_mask = ~img_mask

            acts_img = acts[img_mask.to(device)]   # [n_img, d_sae]
            acts_pmt = acts[pmt_mask.to(device)]   # [n_pmt, d_sae]

            if acts_img.shape[0] == 0 or acts_pmt.shape[0] == 0:
                continue

            v_img = acts_img.max(dim=0).values.cpu().float().numpy()  # [d_sae]
            v_pmt = acts_pmt.max(dim=0).values.cpu().float().numpy()  # [d_sae]

            if selected_img_feats is not None:
                v_img = v_img[selected_img_feats]
            if selected_pmt_feats is not None:
                v_pmt = v_pmt[selected_pmt_feats]

            X_img_list.append(v_img)
            X_pmt_list.append(v_pmt)
            y_list.append(label)

    X_img = np.array(X_img_list, dtype=np.float32)
    X_pmt = np.array(X_pmt_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X_img, X_pmt, y


def _train_bilinear(
    X_img: np.ndarray,
    X_pmt: np.ndarray,
    y: np.ndarray,
    rank: int,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: str,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple["BilinearProbe", float]:
    """Train a BilinearProbe; returns (model, best_val_acc).

    Uses stratified val split and pos_weight-balanced BCE loss to avoid
    collapsing to always-negative predictions on imbalanced data.
    Features are assumed to already be StandardScaler-normalised by the caller.
    """
    rng = np.random.RandomState(seed)

    # Stratified split so val reflects class balance
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_pos_val = max(1, int(len(pos_idx) * val_split))
    n_neg_val = max(1, int(len(neg_idx) * val_split))
    val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
    tr_idx  = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])

    X_img_tr = torch.tensor(X_img[tr_idx], device=device)
    X_pmt_tr = torch.tensor(X_pmt[tr_idx], device=device)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.float32, device=device)

    X_img_va = torch.tensor(X_img[val_idx], device=device)
    X_pmt_va = torch.tensor(X_pmt[val_idx], device=device)
    y_va = torch.tensor(y[val_idx], dtype=torch.float32, device=device)

    # Class-weighted loss: prevents collapse to all-negative on ~54/46 split
    n_pos = float(y_tr.sum().item())
    n_neg = float(len(y_tr)) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    logger.info("Bilinear: n_train=%d n_pos=%.0f n_neg=%.0f pos_weight=%.3f",
                len(tr_idx), n_pos, n_neg, float(pos_weight.item()))

    k_img, k_pmt = X_img.shape[1], X_pmt.shape[1]
    model = BilinearProbe(k_img, k_pmt, rank).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc = 0.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_img_tr, X_pmt_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_img_va, X_pmt_va)
                val_preds = (val_logits > 0).float()
                val_acc = (val_preds == y_va).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stop at epoch %d (val_acc=%.4f)", epoch + 1, best_val_acc)
                    break

            if (epoch + 1) % 50 == 0:
                logger.info("Epoch %d: train_loss=%.4f val_acc=%.4f", epoch + 1, loss.item(), best_val_acc)

    model.load_state_dict(best_state)
    return model, best_val_acc


def _eval_probe(model: "BilinearProbe", X_img: np.ndarray, X_pmt: np.ndarray, y: np.ndarray, device: str) -> dict:
    """Evaluate BilinearProbe; returns accuracy, F1, pred_pos_rate."""
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    X_img_t = torch.tensor(X_img, device=device)
    X_pmt_t = torch.tensor(X_pmt, device=device)
    with torch.no_grad():
        logits = model(X_img_t, X_pmt_t).cpu().numpy()
    preds = (logits > 0).astype(int)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, average="binary", zero_division=0)),
        "pred_pos_rate": float(preds.mean()),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
    }


def _eval_sklearn(X: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """Train and evaluate a logistic regression probe for comparison.

    Uses 'saga' solver which handles high-dimensional sparse data well and
    converges reliably without scaling issues that affect lbfgs.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", solver="saga")
    clf.fit(X, y_train)
    preds = clf.predict(X_val)
    return {
        "accuracy": float(accuracy_score(y_val, preds)),
        "f1": float(f1_score(y_val, preds, average="binary", zero_division=0)),
        "pred_pos_rate": float(preds.mean()),
        "n": int(len(y_val)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FHMBilinearConfig)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(cfg.train_cache_dir)
    val_dir = Path(cfg.val_cache_dir)
    device = cfg.device

    # Load SAE
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    sae.eval()
    residual_stats = load_residual_stats(cfg.residual_sae_path)
    logger.info("Residual SAE loaded: %d latents", sae.num_latents)

    # ── Load train manifest (FHM train split) ────────────────────────────────
    train_manifest = _read_jsonl(train_dir / "manifest.jsonl")
    fhm_train = [
        r for r in train_manifest
        if r.get("dataset_name") == FHM_DATASET
        and r.get("task_name") == FHM_TASK
        and r.get("split") == "train"
        and r.get("gold_fields")
    ]
    logger.info("FHM train rows: %d", len(fhm_train))

    # ── Feature selection: top-K by activation variance on train ─────────────
    logger.info("Extracting full-d_sae features for variance-based selection ...")
    X_img_full, X_pmt_full, y_train = _extract_role_features(
        sae, residual_stats, train_dir, fhm_train,
        selected_img_feats=None, selected_pmt_feats=None, device=device,
    )
    logger.info("Full feature shapes: X_img=%s X_pmt=%s y=%s", X_img_full.shape, X_pmt_full.shape, y_train.shape)

    var_img = X_img_full.var(axis=0)
    var_pmt = X_pmt_full.var(axis=0)
    sel_img = np.argsort(-var_img)[: cfg.top_k_features]
    sel_pmt = np.argsort(-var_pmt)[: cfg.top_k_features]
    logger.info("Selected %d image-role and %d prompt-role features", len(sel_img), len(sel_pmt))

    X_img_tr = X_img_full[:, sel_img]
    X_pmt_tr = X_pmt_full[:, sel_pmt]

    # ── Normalise features (fit on train, apply to eval) ─────────────────────
    # Critical: raw SAE activations span [0, ~10+] with heavy right-skew.
    # Without scaling, lbfgs fails to converge and bilinear collapses to
    # always-negative predictions (bias dominates).
    from sklearn.preprocessing import StandardScaler
    scaler_img = StandardScaler()
    X_img_tr = scaler_img.fit_transform(X_img_tr)
    scaler_pmt = StandardScaler()
    X_pmt_tr = scaler_pmt.fit_transform(X_pmt_tr)
    logger.info("Features normalised: img mean=%.3f std=%.3f | pmt mean=%.3f std=%.3f",
                X_img_tr.mean(), X_img_tr.std(), X_pmt_tr.mean(), X_pmt_tr.std())

    # ── Load val/test manifest (FHM proper eval split) ────────────────────────
    val_manifest = _read_jsonl(val_dir / "manifest.jsonl")
    eval_splits = set(cfg.eval_splits)
    fhm_val = [
        r for r in val_manifest
        if r.get("dataset_name") == FHM_DATASET
        and r.get("task_name") == FHM_TASK
        and r.get("split") in eval_splits
        and r.get("gold_fields")
    ]
    if not fhm_val:
        logger.error(
            "No FHM eval rows found in %s for splits %s. "
            "Run cache_dense_val_test.sub + compute_residuals_val_test.sub first.",
            val_dir, cfg.eval_splits,
        )
        return
    logger.info("FHM eval rows: %d (splits: %s)", len(fhm_val), {r["split"] for r in fhm_val})

    X_img_va, X_pmt_va, y_val = _extract_role_features(
        sae, residual_stats, val_dir, fhm_val,
        selected_img_feats=sel_img, selected_pmt_feats=sel_pmt, device=device,
    )
    logger.info("Eval shapes: X_img=%s X_pmt=%s y=%s", X_img_va.shape, X_pmt_va.shape, y_val.shape)

    # Apply the same scalers fitted on train
    X_img_va = scaler_img.transform(X_img_va)
    X_pmt_va = scaler_pmt.transform(X_pmt_va)

    results: dict = {}

    # ── Baselines: image-only, prompt-only, concat ───────────────────────────
    logger.info("Training image-only probe ...")
    results["image_only"] = _eval_sklearn(X_img_tr, y_train, X_img_va, y_val)

    logger.info("Training prompt-only probe ...")
    results["prompt_only"] = _eval_sklearn(X_pmt_tr, y_train, X_pmt_va, y_val)

    logger.info("Training concat (image + prompt) probe ...")
    X_concat_tr = np.concatenate([X_img_tr, X_pmt_tr], axis=1)
    X_concat_va = np.concatenate([X_img_va, X_pmt_va], axis=1)
    results["concat"] = _eval_sklearn(X_concat_tr, y_train, X_concat_va, y_val)

    # ── Bilinear probe ────────────────────────────────────────────────────────
    logger.info("Training bilinear probe (rank=%d) ...", cfg.bilinear_rank)
    bil_model, best_val_acc = _train_bilinear(
        X_img_tr, X_pmt_tr, y_train,
        rank=cfg.bilinear_rank,
        n_epochs=cfg.n_epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        device=device,
    )
    logger.info("Bilinear training done; internal val_acc=%.4f", best_val_acc)
    results["bilinear"] = _eval_probe(bil_model, X_img_va, X_pmt_va, y_val, device)

    # ── Write results ─────────────────────────────────────────────────────────
    summary_path = output_dir / "fhm_bilinear_results.json"
    with summary_path.open("w") as fd:
        json.dump({
            "config": {
                "sae_checkpoint": cfg.residual_sae_path,
                "top_k_features": cfg.top_k_features,
                "bilinear_rank": cfg.bilinear_rank,
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
            },
            "results": results,
        }, fd, indent=2)

    logger.info("Results written to %s", summary_path)

    # Print summary table
    print(f"\n{'Method':<20} {'acc':>8} {'F1':>8} {'ppr':>8}")
    print("-" * 48)
    for method, m in results.items():
        print(f"  {method:<18} {m['accuracy']:>8.4f} {m['f1']:>8.4f} {m['pred_pos_rate']:>8.4f}")

    # Highlight bilinear gain vs concat
    concat_acc = results.get("concat", {}).get("accuracy", 0)
    bilinear_acc = results.get("bilinear", {}).get("accuracy", 0)
    delta = bilinear_acc - concat_acc
    sign = "+" if delta >= 0 else ""
    print(f"\nBilinear vs concat: {sign}{delta:+.4f} accuracy")
    if delta > 0.005:
        print("→ Bilinear interaction IS informative beyond unimodal + concat.")
    elif delta < -0.005:
        print("→ Bilinear interaction HURTS relative to concat (likely overfitting the low-rank")
        print("  interaction term) — this is a null/negative result, not a gain.")
    else:
        print("→ Bilinear interaction adds little over concat (cross-modal structure weak).")


if __name__ == "__main__":
    main()
