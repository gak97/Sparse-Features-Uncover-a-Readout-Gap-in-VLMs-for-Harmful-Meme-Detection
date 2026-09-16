

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from simple_parsing import parse, Serializable, list_field

from ..qwen import QwenSae
from ..task_labels import (
    BINARY_TASK_DATASET,
    BINARY_TASK_FIELD,
    BINARY_TASKS,
    MULTICLASS_TASKS,
    compute_metrics,
    extract_gold_value,
    fit_scaled_logreg,
)

logger = logging.getLogger(__name__)


@dataclass
class LogitProjectionConfig(Serializable):
    predictions_jsonl: str = ""          # baseline_proper.jsonl (joint_infer.py, apply_residual_sae=false)
    train_cache_dir: str = "./qwen_residual_sae_cache/dense"
    val_cache_dir: str = "./qwen_residual_sae_cache/dense_val_test"
    output_dir: str = "./qwen_logit_projection"
    device: str = "cuda:0"
    aggregation: str = "max"

    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20

    eval_splits: list[str] = list_field("validation", "val", "dev", "test", "test_seen")
    C: float = 1.0
    max_iter: int = 3000
    min_class_samples: int = 20

    alpha_grid: list[float] = list_field(0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    calibration_fraction: float = 0.30
    seed: int = 42


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fd:
        for line in fd:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _extract_features_with_ids(
    base_sae: QwenSae,
    cache_dir: Path,
    manifest_rows: list[dict],
    field: str,
    aggregation: str,
    device: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract base SAE feature vectors + raw gold label strings + sample_ids."""
    by_shard: dict[str, list] = defaultdict(list)
    for row in manifest_rows:
        by_shard[row["shard_file"]].append(row)

    d_sae = base_sae.cfg.d_sae
    X = np.empty((len(manifest_rows), d_sae), dtype=np.float32)
    labels: list[str] = []
    sample_ids: list[str] = []
    n_filled = 0

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

        for row in rows:
            gold_val = extract_gold_value(row.get("gold_fields", {}), field)
            if gold_val is None:
                continue
            offset, length = row["offset"], row["length"]
            sample_h = hidden_all[offset: offset + length]
            if sample_h.shape[0] == 0:
                continue

            with torch.no_grad():
                acts, _ = base_sae.encode(sample_h)
                vec = acts.max(dim=0).values if aggregation == "max" else acts.mean(dim=0)

            X[n_filled] = vec.cpu().float().numpy()
            labels.append(gold_val)
            sample_ids.append(str(row.get("sample_id")))
            n_filled += 1

    X = X[:n_filled]
    return X, labels, sample_ids


def _logit_diff_from_prediction_row(row: dict, positive_label: str, negative_label: str,
                                     positive_group: set[str] | None = None,
                                     negative_group: set[str] | None = None) -> float | None:
    """Recover a signed logit difference (positive => favours positive class)
    from a joint_infer.py prediction row, for either decode strategy."""
    strategy = row.get("decode_strategy")
    if strategy == "yes_no":
        conf_gap = row.get("conf_gap")
        if conf_gap is None:
            return None
        p = min(max(float(conf_gap), 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))
    if strategy == "label_score":
        label_scores = row.get("label_scores")
        if not label_scores:
            return None
        pos_scores = [v for k, v in label_scores.items() if k in (positive_group or {positive_label})]
        neg_scores = [v for k, v in label_scores.items() if k in (negative_group or {negative_label})]
        if not pos_scores or not neg_scores:
            return None
        return float(_logsumexp(pos_scores) - _logsumexp(neg_scores))
    return None


def _logsumexp(values: list[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def _sweep_alpha(logit_diff: np.ndarray, probe_z: np.ndarray, gold_bin: np.ndarray,
                  alpha_grid: list[float]) -> tuple[float, dict]:
    best_alpha, best_f1, best_metrics = 0.0, -1.0, {}
    for alpha in alpha_grid:
        combined = logit_diff + alpha * probe_z
        pred = (combined > 0).astype(int)
        m = compute_metrics(list(gold_bin), list(pred))
        if m["macro_f1"] > best_f1:
            best_alpha, best_f1, best_metrics = alpha, m["macro_f1"], m
    return best_alpha, best_metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(LogitProjectionConfig)

    assert cfg.predictions_jsonl, "--predictions_jsonl is required (baseline_proper.jsonl)"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(cfg.train_cache_dir)
    val_dir = Path(cfg.val_cache_dir)

    base_sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    logger.info("Base SAE loaded: d_model=%d d_sae=%d k=%d", base_sae.cfg.d_model, base_sae.cfg.d_sae, base_sae.cfg.k)

    pred_rows = _read_jsonl(Path(cfg.predictions_jsonl))
    logger.info("Loaded %d prediction rows from %s", len(pred_rows), cfg.predictions_jsonl)

    train_manifest = _read_jsonl(train_dir / "manifest.jsonl")
    train_manifest = [r for r in train_manifest if r.get("split") == "train"]
    eval_splits = set(cfg.eval_splits)
    val_manifest = _read_jsonl(val_dir / "manifest.jsonl")
    val_manifest = [r for r in val_manifest if r.get("split") in eval_splits]

    # Build the unified list of (dataset_name, task_name, field, positive_label,
    # negative_label, positive_group, negative_group) for all six binary views.
    task_specs = []
    for task_name, pos_label in BINARY_TASKS.items():
        ds = BINARY_TASK_DATASET[task_name]
        field = BINARY_TASK_FIELD[task_name]
        # Native binary schemas: negative label is "the other" of exactly 2 labels.
        # We don't have the full label list here, but yes_no decode only needs
        # conf_gap, so pos/neg groups are unused for these tasks.
        task_specs.append((ds, task_name, field, pos_label, None, None, None))
    for task_name, spec in MULTICLASS_TASKS.items():
        pos_group = {c for c in spec.classes if spec.collapse(c) == spec.binary_positive}
        neg_group = {c for c in spec.classes if spec.collapse(c) == spec.binary_negative}
        task_specs.append((
            spec.dataset_name, task_name, spec.field,
            spec.binary_positive, spec.binary_negative, pos_group, neg_group,
        ))

    all_results = []

    for ds, task_name, field, pos_label, neg_label, pos_group, neg_group in task_specs:
        logger.info("=== Task: %s/%s (field=%s) ===", ds, task_name, field)

        train_rows = [r for r in train_manifest if r.get("task_name") == task_name and r.get("gold_fields")]
        val_rows = [r for r in val_manifest if r.get("task_name") == task_name and r.get("gold_fields")]
        if not train_rows or not val_rows:
            logger.info("Missing train/eval rows; skipping")
            continue

        # Fit base SAE probe on train.
        X_tr, y_tr_raw, _ = _extract_features_with_ids(base_sae, train_dir, train_rows, field, cfg.aggregation, cfg.device)
        if pos_group is None:
            y_tr_bin = np.array([1 if v.lower() == pos_label.lower() else 0 for v in y_tr_raw], dtype=np.int32)
        else:
            y_tr_bin = np.array([1 if v in pos_group else 0 for v in y_tr_raw], dtype=np.int32)
        n_pos, n_neg = int(y_tr_bin.sum()), int((y_tr_bin == 0).sum())
        if n_pos < cfg.min_class_samples or n_neg < cfg.min_class_samples:
            logger.info("Too few train class samples; skipping")
            continue
        clf, scaler = fit_scaled_logreg(X_tr, y_tr_bin, C=cfg.C, max_iter=cfg.max_iter)

        # Extract eval features + probe decision_function scores.
        X_va, y_va_raw, sample_ids = _extract_features_with_ids(base_sae, val_dir, val_rows, field, cfg.aggregation, cfg.device)
        if len(X_va) < 20:
            logger.info("Too few eval rows; skipping")
            continue
        if pos_group is None:
            y_va_bin = np.array([1 if v.lower() == pos_label.lower() else 0 for v in y_va_raw], dtype=np.int32)
        else:
            y_va_bin = np.array([1 if v in pos_group else 0 for v in y_va_raw], dtype=np.int32)
        probe_score = clf.decision_function(scaler.transform(X_va))  # [N], raw logit-space score

        # Join with baseline predictions by sample_id.
        pred_by_id = {
            str(r.get("sample_id")): r
            for r in pred_rows
            if r.get("dataset_name") == ds and r.get("task_name") == task_name
        }
        logit_diffs, joined_idx = [], []
        for i, sid in enumerate(sample_ids):
            row = pred_by_id.get(sid)
            if row is None:
                continue
            ld = _logit_diff_from_prediction_row(row, pos_label, neg_label or "", pos_group, neg_group)
            if ld is None:
                continue
            logit_diffs.append(ld)
            joined_idx.append(i)
        if len(joined_idx) < 20:
            logger.info("Too few joined rows (n=%d) between predictions and eval features; skipping", len(joined_idx))
            continue

        logit_diff = np.array(logit_diffs, dtype=np.float64)
        gold_bin = y_va_bin[joined_idx]
        probe_score_j = probe_score[joined_idx]

        # Calibration / final split (fixed seed, disjoint).
        rng = random.Random(cfg.seed)
        idx = list(range(len(joined_idx)))
        rng.shuffle(idx)
        n_cal = max(10, int(len(idx) * cfg.calibration_fraction))
        cal_idx = np.array(idx[:n_cal])
        final_idx = np.array(idx[n_cal:])

        # z-score the probe score using CALIBRATION-set stats only (no leakage).
        mu, sigma = probe_score_j[cal_idx].mean(), probe_score_j[cal_idx].std() + 1e-8
        probe_z = (probe_score_j - mu) / sigma

        best_alpha, cal_metrics = _sweep_alpha(logit_diff[cal_idx], probe_z[cal_idx], gold_bin[cal_idx], cfg.alpha_grid)

        # Final (held-out from calibration) metrics.
        baseline_pred = (logit_diff[final_idx] > 0).astype(int)
        baseline_metrics = compute_metrics(list(gold_bin[final_idx]), list(baseline_pred))

        combined_final = logit_diff[final_idx] + best_alpha * probe_z[final_idx]
        combined_pred = (combined_final > 0).astype(int)
        combined_metrics = compute_metrics(list(gold_bin[final_idx]), list(combined_pred))

        probe_only_pred = (probe_z[final_idx] > 0).astype(int)
        probe_only_metrics = compute_metrics(list(gold_bin[final_idx]), list(probe_only_pred))

        result = {
            "dataset_name": ds,
            "task_name": task_name,
            "n_calibration": int(len(cal_idx)),
            "n_final": int(len(final_idx)),
            "best_alpha": best_alpha,
            "calibration_macro_f1_at_best_alpha": cal_metrics.get("macro_f1"),
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "combined_macro_f1": combined_metrics["macro_f1"],
            "probe_only_macro_f1": probe_only_metrics["macro_f1"],
            "delta_vs_baseline": round(combined_metrics["macro_f1"] - baseline_metrics["macro_f1"], 4),
        }
        all_results.append(result)
        logger.info(
            "  alpha*=%.3g  baseline=%.4f  combined=%.4f  probe_only=%.4f  (Δ=%+.4f)",
            best_alpha, baseline_metrics["macro_f1"], combined_metrics["macro_f1"],
            probe_only_metrics["macro_f1"], result["delta_vs_baseline"],
        )

    results_path = output_dir / "logit_projection_results.jsonl"
    with results_path.open("w") as fd:
        for r in all_results:
            fd.write(json.dumps(r) + "\n")
    logger.info("Results written to %s", results_path)

    print(f"\n{'Task':<45} {'alpha*':>7} {'baseline':>9} {'combined':>9} {'probe_only':>10} {'delta':>8}")
    print("-" * 92)
    for r in all_results:
        key = f"{r['dataset_name']}/{r['task_name']}"
        print(f"  {key:<43} {r['best_alpha']:>7.3g} {r['baseline_macro_f1']:>9.4f} "
              f"{r['combined_macro_f1']:>9.4f} {r['probe_only_macro_f1']:>10.4f} {r['delta_vs_baseline']:>+8.4f}")


if __name__ == "__main__":
    main()
