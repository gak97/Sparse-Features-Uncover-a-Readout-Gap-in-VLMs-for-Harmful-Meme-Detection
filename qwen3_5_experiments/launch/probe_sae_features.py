

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import load_file
from simple_parsing import parse, Serializable, list_field

from gemma3_experiments.residual_sae.normalization import load_residual_stats
from ..models import load_residual_sae
from ..task_labels import MULTICLASS_TASKS, extract_gold_value, fit_scaled_logreg, predict_scaled

logger = logging.getLogger(__name__)

POSITIVE_LABELS = {
    "hateful_memes_binary": "hateful",
    "subtask_a_hate_speech": "Hate Speech",
    "misogyny_and_subtypes": "misogynous",
    "offensive_detection": "offensive",
}


@dataclass
class ProbeSAEConfig(Serializable):
    residual_cache_dir: str = "./qwen_residual_sae_cache/residual"
    residual_sae_path: str = "./qwen_residual_sae_ckpts/best_finite"
    output_dir: str = "./qwen_probe_results"
    device: str = "cuda:0"
    aggregation: str = "max"         # max | mean
    eval_fraction: float = 0.20
    seed: int = 42
    min_class_samples: int = 20
    max_samples_per_task: int | None = None
    splits: list[str] = list_field("train")
    # LR hyperparameters
    C: float = 1.0
    max_iter: int = 3000


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_features(sae, residuals: torch.Tensor, residual_stats, aggregation: str) -> np.ndarray:
    """Extract (n_samples_in_batch, n_latents) as numpy, aggregating over tokens."""
    sae_in = residual_stats.normalize(residuals) if residual_stats is not None else residuals
    acts, _ = sae.encode(sae_in)  # (n_tokens, n_latents)
    if aggregation == "max":
        vec = acts.max(dim=0).values
    else:
        vec = acts.mean(dim=0)
    return vec.cpu().float().numpy()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ProbeSAEConfig)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        from scipy.sparse import lil_matrix, csr_matrix
    except ImportError:
        raise ImportError("scikit-learn and scipy required: pip install scikit-learn scipy")

    cache_dir = Path(cfg.residual_cache_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device

    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    sae.eval()
    residual_stats = load_residual_stats(cfg.residual_sae_path)
    n_latents = sae.num_latents
    logger.info("SAE loaded: %d latents, L0 via discovery from %s", n_latents, cfg.residual_sae_path)

    manifest_rows = _read_jsonl(cache_dir / "manifest.jsonl")
    splits = set(cfg.splits)
    manifest_rows = [r for r in manifest_rows if r.get("split") in splits]
    logger.info("Manifest: %d rows", len(manifest_rows))

    # Group by binary task
    binary_tasks_manifest: dict[tuple, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        task_name = row.get("task_name", "")
        if task_name in POSITIVE_LABELS:
            binary_tasks_manifest[(row["dataset_name"], task_name)].append(row)

    all_probe_results = []
    top_features_out: dict[str, dict] = {}

    for (ds_name, task_name), rows in sorted(binary_tasks_manifest.items()):
        pos_label = POSITIVE_LABELS[task_name]
        # Filter rows with gold labels
        labeled = []
        for row in rows:
            gold = row.get("gold_fields", {})
            if not gold:
                continue
            gold_val = str(list(gold.values())[0])
            label = 1 if gold_val.lower() == pos_label.lower() else 0
            labeled.append((row, label))

        if cfg.max_samples_per_task:
            rng = random.Random(cfg.seed)
            rng.shuffle(labeled)
            labeled = labeled[:cfg.max_samples_per_task]

        n_pos = sum(1 for _, l in labeled if l == 1)
        n_neg = sum(1 for _, l in labeled if l == 0)
        if n_pos < cfg.min_class_samples or n_neg < cfg.min_class_samples:
            logger.info("Task %s/%s: skipping (pos=%d, neg=%d)", ds_name, task_name, n_pos, n_neg)
            continue

        logger.info("Task %s/%s: n_pos=%d n_neg=%d extracting features...", ds_name, task_name, n_pos, n_neg)

        # Stratified split
        rng = random.Random(cfg.seed)
        pos_rows = [(r, l) for r, l in labeled if l == 1]
        neg_rows = [(r, l) for r, l in labeled if l == 0]
        rng.shuffle(pos_rows); rng.shuffle(neg_rows)
        n_pos_eval = max(1, int(len(pos_rows) * cfg.eval_fraction))
        n_neg_eval = max(1, int(len(neg_rows) * cfg.eval_fraction))
        eval_rows = pos_rows[:n_pos_eval] + neg_rows[:n_neg_eval]
        train_rows = pos_rows[n_pos_eval:] + neg_rows[n_neg_eval:]

        # Extract features shard by shard
        def extract_set(row_label_pairs):
            by_shard: dict[str, list] = defaultdict(list)
            for row, label in row_label_pairs:
                by_shard[row["shard_file"]].append((row, label))
            X_list, y_list = [], []
            for shard_name, shard_pairs in sorted(by_shard.items()):
                shard_path = cache_dir / shard_name
                if not shard_path.exists():
                    continue
                try:
                    tensors = load_file(str(shard_path))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", shard_path, e)
                    continue
                residual_all = tensors["residual"].to(device=device, dtype=torch.float32)
                for row, label in shard_pairs:
                    offset, length = row["offset"], row["length"]
                    sample_res = residual_all[offset: offset + length]
                    if sample_res.shape[0] == 0:
                        continue
                    with torch.no_grad():
                        feat_vec = _extract_features(sae, sample_res, residual_stats, cfg.aggregation)
                    X_list.append(feat_vec)
                    y_list.append(label)
            return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

        X_train, y_train = extract_set(train_rows)
        X_eval, y_eval = extract_set(eval_rows)
        logger.info("  X_train=%s X_eval=%s", X_train.shape, X_eval.shape)

        if len(X_train) < 10 or len(X_eval) < 2:
            logger.info("  Skipping: too few samples after extraction.")
            continue

        # Fit logistic regression (MaxAbs-scaled to avoid lbfgs convergence failures)
        clf, scaler = fit_scaled_logreg(X_train, y_train, C=cfg.C, max_iter=cfg.max_iter)
        y_pred = predict_scaled(clf, scaler, X_eval)
        acc = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred, average="binary", zero_division=0)
        macro_f1 = f1_score(y_eval, y_pred, average="macro", zero_division=0)
        report = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)

        logger.info("  acc=%.4f F1=%.4f macro_f1=%.4f (n_eval=%d)", acc, f1, macro_f1, len(y_eval))

        result = {
            "dataset_name": ds_name, "task_name": task_name, "positive_label": pos_label,
            "view": "binary",
            "n_train": len(X_train), "n_eval": len(X_eval),
            "n_train_pos": int(y_train.sum()), "n_eval_pos": int(y_eval.sum()),
            "accuracy": round(float(acc), 4), "f1": round(float(f1), 4),
            "macro_f1": round(float(macro_f1), 4),
            "precision": round(float(report.get("1", {}).get("precision", 0)), 4),
            "recall": round(float(report.get("1", {}).get("recall", 0)), 4),
        }
        all_probe_results.append(result)

        # Top feature weights (coefficient = logistic regression weight for positive class)
        if hasattr(clf, "coef_"):
            coef = clf.coef_[0]  # (n_latents,)
            top_pos_idx = np.argsort(-coef)[:20]
            top_neg_idx = np.argsort(coef)[:20]
            task_key = f"{ds_name}::{task_name}"
            top_features_out[task_key] = {
                "label_up": [{"feature_idx": int(i), "weight": round(float(coef[i]), 4)} for i in top_pos_idx],
                "label_down": [{"feature_idx": int(i), "weight": round(float(coef[i]), 4)} for i in top_neg_idx],
            }

    # ── HarMeme + MMHS150K: fine-grained multi-class AND binary-collapsed ──
    # Ablation-only per the locked evaluation protocol (the base SAE probe is
    # the main table's SAE row). Reuses the same 80/20 train-holdout scheme as
    # the four native binary tasks above, for a fair like-for-like comparison.
    multiclass_manifest: dict[str, list[dict]] = defaultdict(list)
    for row in manifest_rows:
        if row.get("task_name") in MULTICLASS_TASKS:
            multiclass_manifest[row["task_name"]].append(row)

    def extract_set_by_field(row_val_pairs, field_unused=None):
        by_shard: dict[str, list] = defaultdict(list)
        for row, label in row_val_pairs:
            by_shard[row["shard_file"]].append((row, label))
        X_list, y_list = [], []
        for shard_name, shard_pairs in sorted(by_shard.items()):
            shard_path = cache_dir / shard_name
            if not shard_path.exists():
                continue
            try:
                tensors = load_file(str(shard_path))
            except Exception as e:
                logger.warning("Failed to load %s: %s", shard_path, e)
                continue
            residual_all = tensors["residual"].to(device=device, dtype=torch.float32)
            for row, label in shard_pairs:
                offset, length = row["offset"], row["length"]
                sample_res = residual_all[offset: offset + length]
                if sample_res.shape[0] == 0:
                    continue
                with torch.no_grad():
                    feat_vec = _extract_features(sae, sample_res, residual_stats, cfg.aggregation)
                X_list.append(feat_vec)
                y_list.append(label)
        return np.array(X_list, dtype=np.float32), y_list

    for task_name, spec in MULTICLASS_TASKS.items():
        rows = multiclass_manifest.get(task_name, [])
        labeled = []
        for row in rows:
            gold_val = extract_gold_value(row.get("gold_fields", {}), spec.field)
            if gold_val is None:
                continue
            labeled.append((row, gold_val))
        if cfg.max_samples_per_task:
            rng = random.Random(cfg.seed)
            rng.shuffle(labeled)
            labeled = labeled[:cfg.max_samples_per_task]
        if len(labeled) < 2 * cfg.min_class_samples:
            logger.info("Task %s: skipping (n=%d too small)", task_name, len(labeled))
            continue

        logger.info("Task %s: n=%d extracting features (fine-grained + binary)...", task_name, len(labeled))
        rng = random.Random(cfg.seed)
        rng.shuffle(labeled)
        n_eval = max(1, int(len(labeled) * cfg.eval_fraction))
        eval_rows = labeled[:n_eval]
        train_rows_mc = labeled[n_eval:]

        X_train, y_train_raw = extract_set_by_field(train_rows_mc)
        X_eval, y_eval_raw = extract_set_by_field(eval_rows)
        if len(X_train) < 10 or len(X_eval) < 2:
            logger.info("  Skipping: too few samples after extraction.")
            continue

        # (a) Native fine-grained multi-class view.
        clf_fine, scaler_fine = fit_scaled_logreg(X_train, y_train_raw, C=cfg.C, max_iter=cfg.max_iter)
        y_pred_fine = predict_scaled(clf_fine, scaler_fine, X_eval)
        fine_acc = accuracy_score(y_eval_raw, y_pred_fine)
        fine_macro_f1 = f1_score(y_eval_raw, y_pred_fine, average="macro", labels=list(spec.classes), zero_division=0)
        fine_result = {
            "dataset_name": spec.dataset_name, "task_name": task_name, "view": "fine_grained",
            "n_train": len(X_train), "n_eval": len(X_eval),
            "accuracy": round(float(fine_acc), 4), "macro_f1": round(float(fine_macro_f1), 4),
        }
        all_probe_results.append(fine_result)
        logger.info("  [fine_grained] acc=%.4f macro_f1=%.4f (n_eval=%d)", fine_acc, fine_macro_f1, len(X_eval))

        # (b) Binary-collapsed view.
        y_train_bin = [spec.collapse(v) for v in y_train_raw]
        y_eval_bin = [spec.collapse(v) for v in y_eval_raw]
        n_pos = sum(1 for v in y_train_bin if v == spec.binary_positive)
        n_neg = len(y_train_bin) - n_pos
        if n_pos < cfg.min_class_samples or n_neg < cfg.min_class_samples:
            logger.info("  [binary] too few class samples; skipping")
            continue
        clf_bin, scaler_bin = fit_scaled_logreg(X_train, y_train_bin, C=cfg.C, max_iter=cfg.max_iter)
        y_pred_bin = predict_scaled(clf_bin, scaler_bin, X_eval)
        bin_acc = accuracy_score(y_eval_bin, y_pred_bin)
        bin_macro_f1 = f1_score(
            y_eval_bin, y_pred_bin, average="macro",
            labels=[spec.binary_negative, spec.binary_positive], zero_division=0,
        )
        bin_result = {
            "dataset_name": spec.dataset_name, "task_name": task_name + " [binary]", "view": "binary",
            "positive_label": spec.binary_positive,
            "n_train": len(X_train), "n_eval": len(X_eval),
            "accuracy": round(float(bin_acc), 4), "macro_f1": round(float(bin_macro_f1), 4),
        }
        all_probe_results.append(bin_result)
        logger.info("  [binary] acc=%.4f macro_f1=%.4f (pos=%s)", bin_acc, bin_macro_f1, spec.binary_positive)

    # Write outputs
    with (output_dir / "probe_results.jsonl").open("w") as fd:
        for r in all_probe_results:
            fd.write(json.dumps(r) + "\n")
    with (output_dir / "top_features.json").open("w") as fd:
        json.dump(top_features_out, fd, indent=2)

    # Summary
    summary_path = output_dir / "probe_summary.txt"
    with summary_path.open("w") as fd:
        fd.write(f"Probe results ({len(all_probe_results)} tasks)\n")
        fd.write(f"SAE: {cfg.residual_sae_path}\n\n")
        fd.write(f"{'Task':<50} {'view':<13} {'acc':>8} {'macro_f1':>9} {'n_eval':>8}\n")
        fd.write("-"*92 + "\n")
        for r in all_probe_results:
            key = f"{r['dataset_name']}/{r['task_name']}"
            view = r.get("view", "binary")
            macro_f1 = r.get("macro_f1", r.get("f1"))
            fd.write(f"  {key:<48} {view:<13} {r['accuracy']:>8.4f} {macro_f1:>9.4f} {r['n_eval']:>8d}\n")
    logger.info("Probe results written to %s", output_dir)

    # Print to stdout
    with summary_path.open() as fd:
        print(fd.read())


if __name__ == "__main__":
    main()
