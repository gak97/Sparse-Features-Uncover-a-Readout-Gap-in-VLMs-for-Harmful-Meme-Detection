

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from sklearn.preprocessing import MaxAbsScaler
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.launch.residual.probe_fhm_crosscoder import (
    DenseLayerSpec,
    GemmaCrosscoder,
)
from gemma3_experiments.launch.residual.probe_fhm_crosscoder_pairwise import (
    CrosscoderPairwiseReadout,
    FHM_LABELS,
    _extract_features_cc,
)
from gemma3_experiments.launch.residual.build_fhm_interaction_cards import (
    _d_model_from_shard,
    _manifest_rows,
)
from gemma3_experiments.launch.residual.probe_validate import (
    _per_class_metrics,
    _read_jsonl,
)
from gemma3_experiments.residual_sae import resolve_visible_device

logger = logging.getLogger(__name__)


@dataclass
class BilinearCheckpointEvalConfig:
    checkpoint_dir: str = ""         # output_dir from probe_fhm_crosscoder_pairwise (with save_checkpoint=True)

    # Dense activation cache for evaluation (test or any split)
    dense_cache_dir: str = ""
    crosscoder_repo_id: str = "google/gemma-scope-2-12b-it"
    crosscoder_hookpoint: str = "resid_post/layer_31_width_65k_l0_medium"
    layer_name: str = "layer31"

    # Which splits to evaluate (use "test" or "test_seen" for FHM)
    eval_splits: list[str] = field(default_factory=lambda: ["test", "test_seen"])

    # Training split (needed to fit scalers consistently with training)
    # If empty, uses the training data from checkpoint_dir's config (same cache dir)
    train_cache_dir: str = ""

    output_dir: str = ""

    batch_size: int = 64
    chunk_size: int = 64
    shard_cache_size: int = 16
    aggregation: str = "max"
    device: str = "cuda:0"


def run(cfg: BilinearCheckpointEvalConfig) -> None:
    device = resolve_visible_device(cfg.device)
    ckpt_dir = Path(cfg.checkpoint_dir)
    eval_cache_dir = Path(cfg.dense_cache_dir)
    out_dir = Path(cfg.output_dir) if cfg.output_dir else (ckpt_dir / "test_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    assert (ckpt_dir / "model_checkpoint.pt").exists(), "model_checkpoint.pt missing"
    state = torch.load(ckpt_dir / "model_checkpoint.pt", map_location="cpu", weights_only=True)
    feat_indices = json.loads((ckpt_dir / "bilinear_feature_indices.json").read_text())["feature_indices"]
    feat_idx_arr = np.array(feat_indices, dtype=np.int64)
    K = len(feat_indices)

    U = state["bil_U"].numpy()
    V = state["bil_V"].numpy()
    r = U.shape[1]
    bil_scale = float(state["bil_scale"].numpy())

    ckpt_metrics = json.loads((ckpt_dir / "pairwise_probe_metrics.json").read_text())
    threshold = float((ckpt_metrics.get("calibrated_metrics") or {}).get("threshold", 0.5))
    saved_cfg = ckpt_metrics.get("config", {})
    img_tok_sel = saved_cfg.get("bilinear_image_token_selection", "image")
    pmt_tok_sel = saved_cfg.get("bilinear_prompt_token_selection", "prompt")
    n_features_full = ckpt_metrics.get("n_features", 65536)
    logger.info("Checkpoint: r=%d  K=%d  scale=%.4f  threshold=%.3f", r, K, bil_scale, threshold)

    # ── Load crosscoder ────────────────────────────────────────────────────────
    # Use eval cache for d_model if train cache not specified
    ref_cache = eval_cache_dir
    d_model = _d_model_from_shard(ref_cache)
    layer_specs = [DenseLayerSpec(cfg.layer_name, str(ref_cache), str(ref_cache))]
    logger.info("Loading crosscoder: %s  %s", cfg.crosscoder_repo_id, cfg.crosscoder_hookpoint)
    crosscoder = GemmaCrosscoder.load_from_hub(
        cfg.crosscoder_repo_id, cfg.crosscoder_hookpoint,
        n_layers=len(layer_specs), d_model=d_model, device=device,
    )

    # ── Load train rows for scaler fitting ────────────────────────────────────
    # If a separate train_cache_dir was specified, use it; else fall back to
    # the cache used during training (same dir used for eval, filtering by "train" split).
    train_cache = Path(cfg.train_cache_dir) if cfg.train_cache_dir else eval_cache_dir
    train_rows = _manifest_rows(train_cache, ["train"], "Hateful_Memes_Dataset")
    if not train_rows:
        # Try the training dir referenced in the checkpoint config
        training_cache_path = saved_cfg.get("layer22_train_dense_cache_dir") or saved_cfg.get("layer31_train_dense_cache_dir")
        if training_cache_path:
            train_rows = _manifest_rows(Path(training_cache_path), ["train"], "Hateful_Memes_Dataset")
    logger.info("Train rows for scaler: %d", len(train_rows))

    # ── Eval rows ─────────────────────────────────────────────────────────────
    eval_rows = _manifest_rows(eval_cache_dir, cfg.eval_splits, "Hateful_Memes_Dataset")
    logger.info("Eval rows: %d (splits=%s)", len(eval_rows), cfg.eval_splits)

    def _extract(rows, tok_sel):
        return _extract_features_cc(
            [rows], [[r.get("shard_file") and f"{r['dataset_name']}::{r['task_name']}::{r['sample_id']}" for r in rows]],
            layer_specs, crosscoder, tok_sel, cfg.aggregation,
            cfg.batch_size, cfg.chunk_size, cfg.shard_cache_size, device,
        )

    # ── Feature extraction ────────────────────────────────────────────────────
    logger.info("Extracting train image/prompt features for scaler fitting …")
    if train_rows:
        tr_img_keys, X_tr_img = _extract(train_rows, img_tok_sel)
        tr_pmt_keys, X_tr_pmt = _extract(train_rows, pmt_tok_sel)
        tr_all_keys, X_tr_all = _extract(train_rows, "all")
        scaler_img = MaxAbsScaler().fit(X_tr_img)
        scaler_pmt = MaxAbsScaler().fit(X_tr_pmt)
        scaler_all = MaxAbsScaler().fit(X_tr_all)
    else:
        logger.warning("No train rows found — fitting scalers on eval data (may differ from training)")
        ev_img_keys_tmp, X_ev_img_tmp = _extract(eval_rows, img_tok_sel)
        ev_pmt_keys_tmp, X_ev_pmt_tmp = _extract(eval_rows, pmt_tok_sel)
        ev_all_keys_tmp, X_ev_all_tmp = _extract(eval_rows, "all")
        scaler_img = MaxAbsScaler().fit(X_ev_img_tmp)
        scaler_pmt = MaxAbsScaler().fit(X_ev_pmt_tmp)
        scaler_all = MaxAbsScaler().fit(X_ev_all_tmp)

    logger.info("Extracting eval features …")
    ev_img_keys, X_ev_img = _extract(eval_rows, img_tok_sel)
    ev_pmt_keys, X_ev_pmt = _extract(eval_rows, pmt_tok_sel)
    ev_all_keys, X_ev_all = _extract(eval_rows, "all")
    assert ev_img_keys == ev_pmt_keys == ev_all_keys
    eval_keys = ev_img_keys

    X_ev_img_top = scaler_img.transform(X_ev_img)[:, feat_idx_arr].toarray().astype(np.float32)
    X_ev_pmt_top = scaler_pmt.transform(X_ev_pmt)[:, feat_idx_arr].toarray().astype(np.float32)
    X_ev_all_sc  = scaler_all.transform(X_ev_all).toarray().astype(np.float32)

    # ── Load model and run inference ──────────────────────────────────────────
    model = CrosscoderPairwiseReadout(
        n_features=n_features_full,
        hidden_size=int(saved_cfg.get("hidden_size", 128)),
        gate_hidden=int(saved_cfg.get("gate_hidden_size", 32)),
        dropout=0.0,
        bilinear_k=K, bilinear_rank=r,
    )
    model.load_state_dict(state)
    model.eval().to(device)

    X_ev_t   = torch.from_numpy(X_ev_all_sc)
    X_ev_img_t = torch.from_numpy(X_ev_img_top)
    X_ev_pmt_t = torch.from_numpy(X_ev_pmt_top)

    with torch.no_grad():
        logits = model(X_ev_t.to(device), X_ev_img_t.to(device), X_ev_pmt_t.to(device)).cpu().numpy()

    scores = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = ["hateful" if s >= threshold else "not hateful" for s in scores]

    # ── Gold labels ────────────────────────────────────────────────────────────
    eval_manifest = {
        f"{r['dataset_name']}::{r['task_name']}::{r['sample_id']}": r
        for r in eval_rows
    }
    y_eval = []
    for key in eval_keys:
        row = eval_manifest.get(key, {})
        label = (row.get("gold_fields") or {}).get("label", "")
        y_eval.append(label if label in FHM_LABELS else "not hateful")

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = _per_class_metrics(y_eval, preds)
    metrics["n"] = len(y_eval)
    metrics["threshold"] = threshold
    metrics["split"] = cfg.eval_splits
    metrics["checkpoint_dir"] = str(ckpt_dir)
    metrics["calibrated_from"] = "val calib split (20% of validation)"

    logger.info("Test metrics: F1=%.4f  Accuracy=%.4f  n=%d",
                metrics["macro_f1"], metrics["accuracy"], metrics["n"])
    print(f"\nFHM bilinear r={r} checkpoint test evaluation")
    print(f"Checkpoint: {ckpt_dir.name}")
    print(f"Split: {cfg.eval_splits}  n={metrics['n']}  threshold={threshold:.3f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}  Accuracy: {metrics['accuracy']:.4f}")
    per_cls = metrics.get("per_class", {})
    for cls, cm in per_cls.items():
        print(f"  {cls}: P={cm.get('precision', 0):.3f} R={cm.get('recall', 0):.3f} F1={cm.get('f1', 0):.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    (out_dir / "bilinear_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / "bilinear_test_predictions.jsonl").open("w") as fd:
        for key, gold, pred, score in zip(eval_keys, y_eval, preds, scores):
            fd.write(json.dumps({
                "sample_key": key,
                "task_key": "Hateful_Memes_Dataset::hateful_memes_binary::label",
                "gold_label": gold,
                "probe_pred": pred,
                "probe_score": round(float(score), 5),
                "probe_correct": pred == gold,
            }) + "\n")
    logger.info("Wrote test results to %s", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(BilinearCheckpointEvalConfig,
                argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
