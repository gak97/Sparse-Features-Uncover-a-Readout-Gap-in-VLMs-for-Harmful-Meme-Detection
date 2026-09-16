

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from simple_parsing import parse, Serializable, list_field

from ..qwen import QwenSae
from .probe_fhm_bilinear import (
    ROLE_IMAGE,
    ROLE_PROMPT,
    BilinearProbe,
    _eval_probe,
    _eval_sklearn,
    _role_mask,
    _train_bilinear,
)

logger = logging.getLogger(__name__)

FHM_DATASET = "Hateful_Memes_Dataset"
FHM_TASK = "hateful_memes_binary"
FHM_POS_LABEL = "hateful"


@dataclass
class FHMBilinearBaseSaeConfig(Serializable):
    train_cache_dir: str = "./qwen_residual_sae_cache/dense"
    val_cache_dir: str = "./qwen_residual_sae_cache/dense_val_test"
    output_dir: str = "./qwen_fhm_bilinear_base_sae"
    device: str = "cuda:0"

    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20

    # Feature selection
    top_k_features: int = 256        # number of features to select per role

    # Bilinear model
    bilinear_rank: int = 16
    n_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-3
    patience: int = 20

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


def _extract_role_features_base_sae(
    base_sae: QwenSae,
    cache_dir: Path,
    manifest_rows: list[dict],
    selected_img_feats: np.ndarray | None,
    selected_pmt_feats: np.ndarray | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract image-role and prompt-role max-pooled BASE SAE feature vectors
    from the dense hidden-state cache. Mirrors
    probe_fhm_bilinear._extract_role_features but for the base SAE + dense
    cache instead of the residual SAE + residual cache."""
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

        hidden_all = tensors["hidden"].to(device=device, dtype=torch.float32)
        roles_all = tensors.get("token_roles")

        for row in rows:
            offset, length = row["offset"], row["length"]
            sample_h = hidden_all[offset: offset + length]
            if sample_h.shape[0] == 0:
                continue

            gold = row.get("gold_fields", {})
            if not gold:
                continue
            gold_val = str(list(gold.values())[0])
            label = 1 if gold_val.lower() == FHM_POS_LABEL.lower() else 0

            if roles_all is not None:
                roles = roles_all[offset: offset + length]
                img_mask = _role_mask(roles, ROLE_IMAGE)
                pmt_mask = _role_mask(roles, ROLE_PROMPT)
            else:
                mid = length // 2
                img_mask = torch.zeros(length, dtype=torch.bool)
                img_mask[:mid] = True
                pmt_mask = ~img_mask

            sample_h_img = sample_h[img_mask.to(device)]
            sample_h_pmt = sample_h[pmt_mask.to(device)]
            if sample_h_img.shape[0] == 0 or sample_h_pmt.shape[0] == 0:
                continue

            with torch.no_grad():
                acts_img, _ = base_sae.encode(sample_h_img)   # [n_img, d_sae]
                acts_pmt, _ = base_sae.encode(sample_h_pmt)   # [n_pmt, d_sae]

            v_img = acts_img.max(dim=0).values.cpu().float().numpy()
            v_pmt = acts_pmt.max(dim=0).values.cpu().float().numpy()

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(FHMBilinearBaseSaeConfig)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(cfg.train_cache_dir)
    val_dir = Path(cfg.val_cache_dir)
    device = cfg.device

    base_sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=device)
    logger.info("Base SAE loaded: d_model=%d d_sae=%d k=%d", base_sae.cfg.d_model, base_sae.cfg.d_sae, base_sae.cfg.k)

    train_manifest = _read_jsonl(train_dir / "manifest.jsonl")
    fhm_train = [
        r for r in train_manifest
        if r.get("dataset_name") == FHM_DATASET
        and r.get("task_name") == FHM_TASK
        and r.get("split") == "train"
        and r.get("gold_fields")
    ]
    logger.info("FHM train rows: %d", len(fhm_train))

    logger.info("Extracting full-d_sae base SAE features for variance-based selection ...")
    X_img_full, X_pmt_full, y_train = _extract_role_features_base_sae(
        base_sae, train_dir, fhm_train,
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

    from sklearn.preprocessing import StandardScaler
    scaler_img = StandardScaler()
    X_img_tr = scaler_img.fit_transform(X_img_tr)
    scaler_pmt = StandardScaler()
    X_pmt_tr = scaler_pmt.fit_transform(X_pmt_tr)
    logger.info("Features normalised: img mean=%.3f std=%.3f | pmt mean=%.3f std=%.3f",
                X_img_tr.mean(), X_img_tr.std(), X_pmt_tr.mean(), X_pmt_tr.std())

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
            "Run cache_dense_val_test.sub first.",
            val_dir, cfg.eval_splits,
        )
        return
    logger.info("FHM eval rows: %d (splits: %s)", len(fhm_val), {r["split"] for r in fhm_val})

    X_img_va, X_pmt_va, y_val = _extract_role_features_base_sae(
        base_sae, val_dir, fhm_val,
        selected_img_feats=sel_img, selected_pmt_feats=sel_pmt, device=device,
    )
    logger.info("Eval shapes: X_img=%s X_pmt=%s y=%s", X_img_va.shape, X_pmt_va.shape, y_val.shape)

    X_img_va = scaler_img.transform(X_img_va)
    X_pmt_va = scaler_pmt.transform(X_pmt_va)

    results: dict = {}

    logger.info("Training image-only probe ...")
    results["image_only"] = _eval_sklearn(X_img_tr, y_train, X_img_va, y_val)

    logger.info("Training prompt-only probe ...")
    results["prompt_only"] = _eval_sklearn(X_pmt_tr, y_train, X_pmt_va, y_val)

    logger.info("Training concat (image + prompt) probe ...")
    X_concat_tr = np.concatenate([X_img_tr, X_pmt_tr], axis=1)
    X_concat_va = np.concatenate([X_img_va, X_pmt_va], axis=1)
    results["concat"] = _eval_sklearn(X_concat_tr, y_train, X_concat_va, y_val)

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

    summary_path = output_dir / "fhm_bilinear_base_sae_results.json"
    with summary_path.open("w") as fd:
        json.dump({
            "config": {
                "sae_repo_id": cfg.sae_repo_id,
                "sae_layer": cfg.sae_layer,
                "top_k_features": cfg.top_k_features,
                "bilinear_rank": cfg.bilinear_rank,
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
            },
            "results": results,
        }, fd, indent=2)

    logger.info("Results written to %s", summary_path)

    print(f"\n{'Method':<20} {'acc':>8} {'F1':>8} {'ppr':>8}")
    print("-" * 48)
    for method, m in results.items():
        print(f"  {method:<18} {m['accuracy']:>8.4f} {m['f1']:>8.4f} {m['pred_pos_rate']:>8.4f}")

    concat_acc = results.get("concat", {}).get("accuracy", 0)
    bilinear_acc = results.get("bilinear", {}).get("accuracy", 0)
    delta = bilinear_acc - concat_acc
    sign = "+" if delta >= 0 else ""
    print(f"\nBilinear vs concat (BASE SAE): {sign}{delta:+.4f} accuracy")
    if delta > 0.005:
        print("→ Bilinear interaction IS informative beyond unimodal + concat, on the base SAE.")
    elif delta < -0.005:
        print("→ Bilinear interaction HURTS relative to concat on the base SAE (likely overfitting the")
        print("  low-rank interaction term on a small train set) — this is a null/negative result, not a gain.")
    else:
        print("→ Bilinear interaction adds little over concat on the base SAE either — ")
        print("  the null result generalizes beyond the residual SAE.")


if __name__ == "__main__":
    main()
