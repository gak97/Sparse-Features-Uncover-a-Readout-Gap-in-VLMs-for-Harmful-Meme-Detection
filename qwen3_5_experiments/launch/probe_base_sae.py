

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from simple_parsing import parse, Serializable, list_field

from ..qwen import QwenSae
from ..task_labels import MULTICLASS_TASKS, extract_gold_value, fit_scaled_logreg, predict_scaled

logger = logging.getLogger(__name__)

POSITIVE_LABELS = {
    "hateful_memes_binary": "hateful",
    "subtask_a_hate_speech": "Hate Speech",
    "misogyny_and_subtypes": "misogynous",
    "offensive_detection": "offensive",
}


@dataclass
class BaseProbeConfig(Serializable):
    train_cache_dir: str = "./qwen_residual_sae_cache/dense"
    val_cache_dir: str = "./qwen_residual_sae_cache/dense_val_test"
    output_dir: str = "./qwen_probe_base_sae"
    device: str = "cuda:0"
    aggregation: str = "max"       # max | mean

    # Base SAE (public Qwen release)
    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20

    # Eval splits for the val/test cache
    eval_splits: list[str] = list_field("validation", "val", "dev", "test", "test_seen")

    # Probe settings
    C: float = 1.0
    max_iter: int = 3000
    seed: int = 42
    min_class_samples: int = 20

    # If True, also run an 80/20 train-holdout probe on the train cache
    # (matches what probe_sae_features.py reports for residual SAE)
    run_train_holdout: bool = True
    eval_fraction: float = 0.20


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fd:
        for line in fd:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _extract_base_sae_features(
    base_sae: QwenSae,
    cache_dir: Path,
    manifest_rows: list[dict],
    aggregation: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract max/mean-pooled base SAE feature vectors from dense hidden cache.

    Returns (X [N, d_sae], y [N]) for labeled rows.
    """
    by_shard: dict[str, list] = defaultdict(list)
    for row in manifest_rows:
        by_shard[row["shard_file"]].append(row)

    X_list: list[np.ndarray] = []
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

        hidden_all = tensors["hidden"].to(device=device, dtype=torch.float32)  # [T, d_model]

        for row in rows:
            # Gold label check
            gold = row.get("gold_fields", {})
            if not gold:
                continue
            task_name = row.get("task_name", "")
            pos_label = POSITIVE_LABELS.get(task_name)
            if pos_label is None:
                continue
            gold_val = str(list(gold.values())[0])
            label = 1 if gold_val.lower() == pos_label.lower() else 0

            offset, length = row["offset"], row["length"]
            sample_h = hidden_all[offset: offset + length]
            if sample_h.shape[0] == 0:
                continue

            # Apply base SAE
            with torch.no_grad():
                acts, _ = base_sae.encode(sample_h)  # [L, d_sae]
                if aggregation == "max":
                    vec = acts.max(dim=0).values
                else:
                    vec = acts.mean(dim=0)

            X_list.append(vec.cpu().float().numpy())
            y_list.append(label)

    X = np.array(X_list, dtype=np.float32) if X_list else np.zeros((0, base_sae.cfg.d_sae), np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def _extract_base_sae_features_by_field(
    base_sae: QwenSae,
    cache_dir: Path,
    manifest_rows: list[dict],
    field: str,
    aggregation: str,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    pass
    by_shard: dict[str, list] = defaultdict(list)
    for row in manifest_rows:
        by_shard[row["shard_file"]].append(row)

    d_sae = base_sae.cfg.d_sae
    X = np.empty((len(manifest_rows), d_sae), dtype=np.float32)
    labels: list[str] = []
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
            n_filled += 1

    X = X[:n_filled]
    return X, labels


def _run_probe(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    C: float, max_iter: int,
):
    from sklearn.metrics import accuracy_score, f1_score

    clf, scaler = fit_scaled_logreg(X_tr, y_tr, C=C, max_iter=max_iter)
    y_pred = predict_scaled(clf, scaler, X_va)
    acc = accuracy_score(y_va, y_pred)
    f1 = f1_score(y_va, y_pred, average="binary", zero_division=0)
    macro_f1 = f1_score(y_va, y_pred, average="macro", zero_division=0)
    result = {
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "pred_pos_rate": round(float(y_pred.mean()), 4),
        "n_train": int(len(y_tr)),
        "n_eval": int(len(y_va)),
        "n_train_pos": int(y_tr.sum()),
        "n_eval_pos": int(y_va.sum()),
    }
    return result, clf


def _run_probe_strings(
    X_tr: np.ndarray, y_tr: list[str],
    X_va: np.ndarray, y_va: list[str],
    C: float, max_iter: int,
    labels: list[str] | None = None,
):
    """Logistic-regression probe over string labels (binary-collapsed or
    native multi-class). Always reports macro-F1 per the locked protocol."""
    from sklearn.metrics import accuracy_score, f1_score

    clf, scaler = fit_scaled_logreg(X_tr, y_tr, C=C, max_iter=max_iter)
    y_pred = predict_scaled(clf, scaler, X_va)
    acc = accuracy_score(y_va, y_pred)
    macro_f1 = f1_score(y_va, y_pred, average="macro", labels=labels, zero_division=0)
    result = {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "n_train": int(len(y_tr)),
        "n_eval": int(len(y_va)),
    }
    return result, clf


def _top_features_from_binary_clf(clf, positive_label, top_k: int = 50) -> dict:
    """Extract top positive/negative-weighted feature indices from a fitted
    binary LogisticRegression (2 classes). `positive_label` selects which
    class the returned `label_up` direction should point toward (1 for
    int-labelled clfs, or a class-name string for string-labelled clfs)."""
    if not hasattr(clf, "coef_"):
        return {}
    coef = clf.coef_[0].copy()
    classes = list(clf.classes_)
    # sklearn's binary coef_ points toward classes_[1]; flip if our "positive"
    # class is actually classes_[0].
    if len(classes) == 2 and classes[1] != positive_label:
        coef = -coef
    top_pos_idx = np.argsort(-coef)[:top_k]
    top_neg_idx = np.argsort(coef)[:top_k]
    return {
        "label_up": [{"feature_idx": int(i), "weight": round(float(coef[i]), 4)} for i in top_pos_idx],
        "label_down": [{"feature_idx": int(i), "weight": round(float(coef[i]), 4)} for i in top_neg_idx],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(BaseProbeConfig)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(cfg.train_cache_dir)
    val_dir = Path(cfg.val_cache_dir)

    # Load base SAE
    logger.info("Loading base Qwen SAE from %s layer %d", cfg.sae_repo_id, cfg.sae_layer)
    base_sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    logger.info("Base SAE loaded: d_model=%d d_sae=%d k=%d", base_sae.cfg.d_model, base_sae.cfg.d_sae, base_sae.cfg.k)

    # Load train manifest
    train_manifest = _read_jsonl(train_dir / "manifest.jsonl")
    train_manifest = [r for r in train_manifest if r.get("split") == "train"]
    logger.info("Train manifest: %d rows", len(train_manifest))

    # Load val/test manifest
    eval_splits = set(cfg.eval_splits)
    val_manifest = _read_jsonl(val_dir / "manifest.jsonl")
    val_manifest = [r for r in val_manifest if r.get("split") in eval_splits]
    logger.info("Eval manifest: %d rows (splits: %s)", len(val_manifest), {r.get("split") for r in val_manifest})

    all_results: list[dict] = []
    top_features_out: dict[str, dict] = {}

    for task_name, pos_label in POSITIVE_LABELS.items():
        logger.info("=== Task: %s (positive=%s) ===", task_name, pos_label)

        train_rows = [r for r in train_manifest if r.get("task_name") == task_name and r.get("gold_fields")]
        val_rows   = [r for r in val_manifest   if r.get("task_name") == task_name and r.get("gold_fields")]

        if not train_rows:
            logger.info("No train rows for task %s; skipping", task_name)
            continue

        # Extract features
        logger.info("Extracting train features (%d rows)...", len(train_rows))
        X_tr, y_tr = _extract_base_sae_features(base_sae, train_dir, train_rows, cfg.aggregation, cfg.device)
        logger.info("X_train shape: %s  pos=%d neg=%d", X_tr.shape, int(y_tr.sum()), int((y_tr==0).sum()))

        if int(y_tr.sum()) < cfg.min_class_samples or int((y_tr == 0).sum()) < cfg.min_class_samples:
            logger.info("Too few class samples; skipping")
            continue

        dataset_name = train_rows[0].get("dataset_name", "?")

        # ── Proper held-out evaluation (train → val/test) ──
        if val_rows:
            logger.info("Extracting eval features (%d rows)...", len(val_rows))
            X_va, y_va = _extract_base_sae_features(base_sae, val_dir, val_rows, cfg.aggregation, cfg.device)
            if len(X_va) >= 10:
                result_proper, clf_proper = _run_probe(X_tr, y_tr, X_va, y_va, cfg.C, cfg.max_iter)
                result_proper["split"] = "proper"
                result_proper["view"] = "binary"
                result_proper["dataset_name"] = dataset_name
                result_proper["task_name"] = task_name
                all_results.append(result_proper)
                top_features_out[f"{dataset_name}::{task_name}"] = _top_features_from_binary_clf(clf_proper, positive_label=1)
                logger.info("  proper eval: acc=%.4f F1=%.4f macro_f1=%.4f", result_proper["accuracy"], result_proper["f1"], result_proper["macro_f1"])
            else:
                logger.info("  Skipping proper eval: too few val rows after extraction")

        # ── Train holdout (80/20, matches probe_sae_features baseline) ──
        if cfg.run_train_holdout and len(X_tr) >= 50:
            rng = random.Random(cfg.seed)
            idx = list(range(len(X_tr)))
            rng.shuffle(idx)
            n_eval = max(1, int(len(idx) * cfg.eval_fraction))
            eval_idx = idx[:n_eval]
            train_idx = idx[n_eval:]
            result_holdout = _run_probe(
                X_tr[train_idx], y_tr[train_idx],
                X_tr[eval_idx], y_tr[eval_idx],
                cfg.C, cfg.max_iter,
            )[0]
            result_holdout["split"] = "train_holdout"
            result_holdout["dataset_name"] = dataset_name
            result_holdout["task_name"] = task_name
            all_results.append(result_holdout)
            logger.info("  train holdout: acc=%.4f F1=%.4f", result_holdout["accuracy"], result_holdout["f1"])

    for task_name, spec in MULTICLASS_TASKS.items():
        logger.info("=== Task: %s (multi-class, field=%s) ===", task_name, spec.field)

        train_rows = [r for r in train_manifest if r.get("task_name") == task_name and r.get("gold_fields")]
        val_rows = [r for r in val_manifest if r.get("task_name") == task_name and r.get("gold_fields")]

        if not train_rows:
            logger.info("No train rows for task %s; skipping", task_name)
            continue

        logger.info("Extracting train features (%d rows)...", len(train_rows))
        X_tr, y_tr_raw = _extract_base_sae_features_by_field(
            base_sae, train_dir, train_rows, spec.field, cfg.aggregation, cfg.device
        )
        if len(X_tr) < 2 * cfg.min_class_samples:
            logger.info("Too few train rows after extraction; skipping")
            continue

        if not val_rows:
            logger.info("No eval rows for task %s; skipping proper eval", task_name)
            continue

        logger.info("Extracting eval features (%d rows)...", len(val_rows))
        X_va, y_va_raw = _extract_base_sae_features_by_field(
            base_sae, val_dir, val_rows, spec.field, cfg.aggregation, cfg.device
        )
        if len(X_va) < 10:
            logger.info("Too few eval rows after extraction; skipping")
            continue

        # (a) Native fine-grained multi-class view — reported SEPARATELY, never
        #     mixed into the binary headline claim. (No top_features saved for
        #     this view — feature cards/alignment are scoped to binary tasks.)
        fine_result, _clf_fine = _run_probe_strings(
            X_tr, y_tr_raw, X_va, y_va_raw, cfg.C, cfg.max_iter, labels=list(spec.classes)
        )
        fine_result["split"] = "proper"
        fine_result["view"] = "fine_grained"
        fine_result["dataset_name"] = spec.dataset_name
        fine_result["task_name"] = task_name
        all_results.append(fine_result)
        logger.info(
            "  [fine_grained] proper eval: acc=%.4f macro_f1=%.4f (%d classes)",
            fine_result["accuracy"], fine_result["macro_f1"], len(spec.classes),
        )

        # (b) Binary-collapsed view — this is the row that belongs in the
        #     headline "binary hate detection" table.
        y_tr_bin = [spec.collapse(v) for v in y_tr_raw]
        y_va_bin = [spec.collapse(v) for v in y_va_raw]
        n_pos = sum(1 for v in y_tr_bin if v == spec.binary_positive)
        n_neg = len(y_tr_bin) - n_pos
        if n_pos < cfg.min_class_samples or n_neg < cfg.min_class_samples:
            logger.info("  [binary] too few class samples in train; skipping")
            continue
        bin_result, clf_bin = _run_probe_strings(
            X_tr, y_tr_bin, X_va, y_va_bin, cfg.C, cfg.max_iter,
            labels=[spec.binary_negative, spec.binary_positive],
        )
        bin_result["split"] = "proper"
        bin_result["view"] = "binary"
        bin_result["dataset_name"] = spec.dataset_name
        bin_result["task_name"] = task_name + " [binary]"
        bin_result["positive_label"] = spec.binary_positive
        all_results.append(bin_result)
        top_features_out[f"{spec.dataset_name}::{task_name} [binary]"] = _top_features_from_binary_clf(
            clf_bin, positive_label=spec.binary_positive
        )
        logger.info(
            "  [binary] proper eval: acc=%.4f macro_f1=%.4f (pos=%s)",
            bin_result["accuracy"], bin_result["macro_f1"], spec.binary_positive,
        )

    # Write results
    results_path = output_dir / "probe_base_sae_results.jsonl"
    with results_path.open("w") as fd:
        for r in all_results:
            fd.write(json.dumps(r) + "\n")
    logger.info("Results written to %s", results_path)

    # Write top_features.json (binary tasks only) — used by base_sae_feature_cards.py
    # and analyze_feature_logit_alignment.py --sae_source base.
    top_features_path = output_dir / "top_features.json"
    with top_features_path.open("w") as fd:
        json.dump(top_features_out, fd, indent=2)
    logger.info("Top features written to %s", top_features_path)

    # Summary table
    print(f"\nBase SAE probe results (sae={cfg.sae_repo_id})")
    print(f"{'Dataset/Task':<55} {'view':<13} {'split':<14} {'acc':>8} {'macro_f1':>9} {'n_eval':>8}")
    print("-" * 112)
    for r in all_results:
        key = f"{r['dataset_name']}/{r['task_name']}"
        view = r.get("view", "binary")
        macro_f1 = r.get("macro_f1", r.get("f1"))
        print(f"  {key:<53} {view:<13} {r['split']:<14} {r['accuracy']:>8.4f} {macro_f1:>9.4f} {r['n_eval']:>8d}")


if __name__ == "__main__":
    main()
