

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.residual_sae import (
    ROLE_GENERATED,
    ROLE_IMAGE,
    ROLE_PROMPT,
    ResidualFeatureDiscoveryConfig,
    load_residual_sae,
    load_residual_stats,
    resolve_visible_device,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────


@dataclass
class ProbeSAEConfig:
    residual_sae_path: str = ""
    residual_cache_dir: str = ""
    output_dir: str = ""

    # Which datasets/splits to include. Only train split is normally cached.
    datasets: list[str] = field(default_factory=lambda: [
        "MMHS150K", "HarMeme", "Hateful_Memes_Dataset", "MAMI", "MultiOFF", "Crisis_HateMM"
    ])
    splits: list[str] = field(default_factory=lambda: ["train"])

    # Fraction of data to hold out for probe evaluation (stratified by label).
    eval_fraction: float = 0.20
    seed: int = 42

    # Feature aggregation (must match how discovery was run).
    token_selection: str = "image"  # image | all | generated | image_and_generated
    aggregation: str = "max"        # max | mean
    batch_size: int = 256

    # Minimum samples per class before including that task in probing.
    min_class_samples: int = 10

    # Logistic regression hyper-parameters.
    # sklearn 1.8+ removed penalty/n_jobs params; use C and l1_ratio instead.
    C: float = 1.0          # inverse regularization strength (smaller = stronger L2)
    max_iter: int = 5000    # more iterations for convergence on large sparse problems
    solver: str = "saga"    # saga: sparse-friendly; lbfgs: faster but needs scaled data
    class_weight: str = "balanced"  # balanced | None — compensates for label imbalance
    # Auto-select liblinear solver for large tasks (exact L2 coordinate descent, much
    # faster than saga/lbfgs on n_train > threshold and avoids convergence issues).
    large_dataset_solver_threshold: int = 20_000

    # For multi-class tasks containing a class named exactly "NotHate", also run a
    # collapsed binary probe (NotHate vs Hate) as an additional task.
    binary_collapse_nothate: bool = True

    # For HarMeme-style 3-class harmfulness tasks, also run a binary collapse:
    # "somewhat harmful" + "very harmful" → "harmful" vs "not harmful".
    binary_collapse_harmful: bool = True

    # Top-k SAE features to report per class (by |weight|).
    top_k_features: int = 20

    # Save fitted probe models as joblib files (for reuse in probe_validate.py).
    save_probes: bool = True

    # Generate matplotlib visualizations at end of run.
    plot: bool = True

    device: str = "cuda:0"


# ──────────────────────────────────────────────────────────────
# Utility — shared with discover_features.py
# ──────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _role_mask(roles: torch.Tensor, token_selection: str) -> torch.Tensor:
    if token_selection == "all":
        return torch.ones_like(roles, dtype=torch.bool)
    if token_selection == "image":
        return roles == ROLE_IMAGE
    if token_selection == "generated":
        return roles == ROLE_GENERATED
    if token_selection == "image_and_generated":
        return (roles == ROLE_IMAGE) | (roles == ROLE_GENERATED)
    if token_selection == "prompt":
        return roles == ROLE_PROMPT
    raise ValueError(f"Unsupported token_selection: {token_selection!r}")


def _row_task_key(row: dict[str, Any]) -> str:
    primary = row["schema"]["primary_field"]
    return f"{row['dataset_name']}::{row['task_name']}::{primary['name']}"


def _row_gold_label(row: dict[str, Any]) -> str | None:
    primary = row["schema"]["primary_field"]
    value = row.get("gold_fields", {}).get(primary["name"])
    return value if isinstance(value, str) else None


def _dense_acts(sae, x: torch.Tensor) -> torch.Tensor:
    if hasattr(sae, "threshold") and hasattr(sae, "pre_acts"):
        pre_acts = sae.pre_acts(x)
        threshold = sae.threshold.to(device=pre_acts.device, dtype=pre_acts.dtype)
        return torch.relu(pre_acts) * (pre_acts > threshold).to(pre_acts.dtype)
    acts, indices = sae.encode(x)
    if indices is None:
        return acts
    dense = x.new_zeros((*acts.shape[:-1], sae.num_latents))
    dense.scatter_(-1, indices, acts)
    return dense


def _aggregate_sample(
    sae,
    residual: torch.Tensor,
    token_roles: torch.Tensor,
    *,
    stats,
    token_selection: str,
    batch_size: int,
    device: str,
    aggregation: str,
) -> torch.Tensor | None:
    mask = _role_mask(token_roles, token_selection)
    if not bool(mask.any()):
        return None
    selected = residual[mask].to(device=device, dtype=torch.float32)
    if stats is not None:
        selected = stats.normalize(selected)

    if aggregation == "max":
        agg = torch.full((sae.num_latents,), -torch.inf, dtype=torch.float32, device=device)
        with torch.inference_mode():
            for start in range(0, selected.shape[0], batch_size):
                acts = _dense_acts(sae, selected[start: start + batch_size])
                agg = torch.maximum(agg, acts.max(dim=0).values.float())
        agg = torch.where(torch.isfinite(agg), agg, torch.zeros_like(agg))
        return agg.cpu()

    if aggregation == "mean":
        total = torch.zeros((sae.num_latents,), dtype=torch.float32, device=device)
        n_tokens = 0
        with torch.inference_mode():
            for start in range(0, selected.shape[0], batch_size):
                acts = _dense_acts(sae, selected[start: start + batch_size])
                total += acts.float().sum(dim=0)
                n_tokens += acts.shape[0]
        assert n_tokens > 0
        return (total / n_tokens).cpu()

    raise ValueError(f"Unsupported aggregation: {aggregation!r}")


# ──────────────────────────────────────────────────────────────
# Stratified split
# ──────────────────────────────────────────────────────────────


def _stratified_split(
    rows: list[dict[str, Any]],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Return (train_ids, eval_ids) stratified by (dataset, task, label)."""
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str, str | None], list[str]] = defaultdict(list)
    for row in rows:
        stratum = (row["dataset_name"], row["task_name"], _row_gold_label(row))
        key = f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"
        by_stratum[stratum].append(key)

    train_ids: set[str] = set()
    eval_ids: set[str] = set()
    for stratum_keys in by_stratum.values():
        rng.shuffle(stratum_keys)
        n_eval = max(1, int(len(stratum_keys) * eval_fraction))
        eval_ids.update(stratum_keys[:n_eval])
        train_ids.update(stratum_keys[n_eval:])
    return train_ids, eval_ids


# ──────────────────────────────────────────────────────────────
# Probe evaluation helpers
# ──────────────────────────────────────────────────────────────


def _per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, Any] = {}
    macro_p = macro_r = macro_f1 = 0.0
    for lbl in labels:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(y_true, y_pred) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g == lbl and p != lbl)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_class[lbl] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}
        macro_p += p; macro_r += r; macro_f1 += f
    n = len(labels)
    correct = sum(1 for g, p in zip(y_true, y_pred) if g == p)
    return {
        "n": len(y_true),
        "accuracy": round(correct / len(y_true), 4) if y_true else 0.0,
        "macro_precision": round(macro_p / n, 4) if n else 0.0,
        "macro_recall": round(macro_r / n, 4) if n else 0.0,
        "macro_f1": round(macro_f1 / n, 4) if n else 0.0,
        "per_class": per_class,
        "gold_counts": dict(Counter(y_true).most_common()),
        "pred_counts": dict(Counter(y_pred).most_common()),
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def run_probe(cfg: ProbeSAEConfig) -> None:
    import gc
    import numpy as np

    try:
        from scipy.sparse import coo_matrix
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
    except ImportError as e:
        raise ImportError(
            "scipy and scikit-learn are required for probing. "
            "Install with: pip install scikit-learn scipy"
        ) from e

    device = resolve_visible_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear output files at the start of each run (avoid accumulating duplicates).
    (output_dir / "probe_results.jsonl").write_text("")
    cache_dir = Path(cfg.residual_cache_dir)

    logger.info("Loading manifest from %s", cache_dir)
    all_rows = _read_jsonl(cache_dir / "manifest.jsonl")
    datasets = set(cfg.datasets)
    splits = set(cfg.splits)
    rows = [
        r for r in all_rows
        if r.get("dataset_name") in datasets
        and r.get("split") in splits
        and _row_gold_label(r) is not None
    ]
    logger.info("Filtered manifest: %d rows across %s datasets", len(rows), list(datasets))
    assert rows, f"No manifest rows matched config from {cache_dir}"

    train_ids, eval_ids = _stratified_split(rows, eval_fraction=cfg.eval_fraction, seed=cfg.seed)
    logger.info("Split: %d train / %d eval", len(train_ids), len(eval_ids))

    def _row_id(row: dict[str, Any]) -> str:
        return f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"

    logger.info("Loading SAE from %s", cfg.residual_sae_path)
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)
    n_features = sae.num_latents
    logger.info("SAE num_latents=%d", n_features)

    # Memory-efficient COO accumulation: store nnz data as numpy arrays, not Python tuples.
    # Python list-of-list-of-tuples overhead: ~6 GB for 160k samples x 300 nnz.
    # numpy COO arrays: ~0.5 GB for the same data (20x reduction).
    from scipy.sparse import coo_matrix

    # COO triplets accumulated across all samples; final shape: (n_samples, n_features)
    coo_rows: list[Any] = []   # np.ndarray int32 per sample (each is a constant-fill array)
    coo_cols: list[Any] = []   # np.ndarray int32 per sample (nnz column indices)
    coo_vals: list[Any] = []   # np.ndarray float32 per sample (nnz values)

    # Per-sample metadata (pure Python lists of short strings — minimal overhead)
    meta_task_key: list[str] = []
    meta_label: list[str] = []
    meta_split: list[str] = []   # 'train' or 'eval'

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)
    del all_rows, rows  # free manifest dicts; only by_shard needed from here

    n_extracted = 0
    for shard_name, shard_rows in sorted(by_shard.items()):
        shard = load_file(str(cache_dir / shard_name))
        residual = shard["residual"]
        token_roles = shard["token_roles"]
        for row in sorted(shard_rows, key=lambda r: int(r["offset"])):
            offset, length = int(row["offset"]), int(row["length"])
            label = _row_gold_label(row)
            if label is None:
                continue
            agg = _aggregate_sample(
                sae,
                residual[offset: offset + length],
                token_roles[offset: offset + length],
                stats=stats,
                token_selection=cfg.token_selection,
                batch_size=cfg.batch_size,
                device=device,
                aggregation=cfg.aggregation,
            )
            if agg is None:
                continue
            rid = _row_id(row)
            task_key = _row_task_key(row)
            sample_idx = n_extracted
            # Store nnz data as numpy arrays — avoids Python object overhead
            nz_mask = agg != 0
            col_arr = nz_mask.nonzero(as_tuple=True)[0].to(torch.int32).numpy()
            val_arr = agg[nz_mask].numpy().astype("float32")
            row_arr = np.full(col_arr.shape[0], sample_idx, dtype=np.int32)
            coo_rows.append(row_arr)
            coo_cols.append(col_arr)
            coo_vals.append(val_arr)
            meta_task_key.append(task_key)
            meta_label.append(label)
            meta_split.append("eval" if rid in eval_ids else "train")
            n_extracted += 1
        del shard, residual, token_roles
        gc.collect()
        logger.info("Shard %s: extracted %d total", shard_name, n_extracted)

    n_total = n_extracted
    logger.info("Total extracted: %d samples", n_total)

    # Build one global sparse CSR matrix (scipy), then slice by task.
    # Peak memory: ~0.5 GB for the COO data; ~0.6 GB for CSR.
    logger.info("Building global CSR matrix (%d x %d)", n_total, n_features)
    all_rows_np = np.concatenate(coo_rows)
    all_cols_np = np.concatenate(coo_cols)
    all_vals_np = np.concatenate(coo_vals)
    del coo_rows, coo_cols, coo_vals   # free the list of arrays

    full_csr = coo_matrix(
        (all_vals_np, (all_rows_np, all_cols_np)),
        shape=(n_total, n_features),
    ).tocsr()
    del all_rows_np, all_cols_np, all_vals_np
    logger.info("CSR matrix built: nnz=%d", full_csr.nnz)

    meta_task_key_arr = np.array(meta_task_key)
    meta_split_arr = np.array(meta_split)

    all_task_keys = sorted(set(meta_task_key))
    probe_results = []
    top_features_all: dict[str, Any] = {}

    for task_key in all_task_keys:
        ds, task, field_name = task_key.split("::", maxsplit=2)

        task_mask = meta_task_key_arr == task_key
        t_mask = task_mask & (meta_split_arr == "train")
        e_mask = task_mask & (meta_split_arr == "eval")
        t_indices = np.where(t_mask)[0]
        e_indices = np.where(e_mask)[0]

        if len(t_indices) == 0 or len(e_indices) == 0:
            logger.warning("Skipping task %s: no train or eval samples", task_key)
            continue

        n_train = len(t_indices)
        n_eval = len(e_indices)

        t_labels_seq = [meta_label[i] for i in t_indices]
        e_labels_seq = [meta_label[i] for i in e_indices]

        # Check minimum class samples
        t_counts = Counter(t_labels_seq)
        if any(c < cfg.min_class_samples for c in t_counts.values()):
            logger.warning(
                "Skipping task %s: some classes have fewer than %d train samples: %s",
                task_key, cfg.min_class_samples, dict(t_counts)
            )
            continue

        logger.info("Slicing CSR for %s (train=%d, eval=%d)", task_key, n_train, n_eval)
        # Slicing a CSR matrix by row indices is fast (O(nnz in selected rows))
        X_train = full_csr[t_indices]
        X_eval = full_csr[e_indices]

        # MaxAbsScaler: sparse-safe (no centering), brings all features to [-1, 1].
        # Critical for lbfgs convergence and helpful for saga on high-magnitude SAE activations.
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)

        le = LabelEncoder()
        y_train = le.fit_transform(t_labels_seq)
        y_eval = le.transform(e_labels_seq)
        classes = le.classes_.tolist()

        # Auto-select solver: liblinear is exact L2 coordinate descent — much faster
        # than saga for large n_train and avoids convergence issues entirely.
        effective_solver = (
            "liblinear"
            if n_train > cfg.large_dataset_solver_threshold
            else cfg.solver
        )

        def _fit_and_record(X_tr, X_ev, y_tr, y_ev, label_encoder, label_seq_ev, task_k: str, extra_tag: str = "") -> None:
            cls_list = label_encoder.classes_.tolist()
            tag = f"{task_k}{extra_tag}"
            logger.info(
                "Fitting LR probe for %s: %d classes=%s, C=%g, class_weight=%s, solver=%s",
                tag, len(cls_list), cls_list, cfg.C, cfg.class_weight, effective_solver
            )
            clf = LogisticRegression(
                C=cfg.C,
                max_iter=cfg.max_iter,
                solver=effective_solver,
                class_weight=cfg.class_weight if cfg.class_weight != "None" else None,
            )
            # liblinear doesn't support multiclass natively — wrap with OVR
            if effective_solver == "liblinear" and len(cls_list) >= 3:
                from sklearn.multiclass import OneVsRestClassifier
                clf = OneVsRestClassifier(clf)
            clf.fit(X_tr, y_tr)
            y_pred_enc = clf.predict(X_ev)
            y_pred = label_encoder.inverse_transform(y_pred_enc).tolist()

            metrics = _per_class_metrics(label_seq_ev, y_pred)
            logger.info("Task %s: accuracy=%.4f macro_f1=%.4f", tag, metrics["accuracy"], metrics["macro_f1"])

            top_feats: dict[str, list[dict[str, Any]]] = {}
            # OneVsRestClassifier wraps N binary estimators; extract coef_ from each
            if hasattr(clf, "estimators_"):
                for ci, (cls_name, est) in enumerate(zip(cls_list, clf.estimators_)):
                    if hasattr(est, "coef_"):
                        coef_row = np.asarray(est.coef_).ravel()
                        abs_w = np.abs(coef_row)
                        top_idx = abs_w.argsort()[::-1][: cfg.top_k_features].tolist()
                        top_feats[cls_name] = [
                            {"feature_idx": int(idx), "weight": round(float(coef_row[idx]), 6)}
                            for idx in top_idx
                        ]
            elif hasattr(clf, "coef_"):
                coef = clf.coef_
                is_binary = coef.shape[0] == 1
                for ci, cls_name in enumerate(cls_list):
                    if is_binary:
                        # sklearn stores one row for binary LR: coef[0] is the weight
                        # vector for classes_[1] (positive class).  Negate for class 0.
                        sign = 1.0 if ci == 1 else -1.0
                        coef_row = coef[0] * sign
                    else:
                        coef_row = coef[ci]
                    abs_w = np.abs(coef_row)
                    top_idx = abs_w.argsort()[::-1][: cfg.top_k_features].tolist()
                    top_feats[cls_name] = [
                        {"feature_idx": int(idx), "weight": round(float(coef_row[idx]), 6)}
                        for idx in top_idx
                    ]
            top_features_all[tag] = top_feats

            # Optionally save the fitted probe model + scaler for probe_validate.py
            if cfg.save_probes:
                try:
                    import joblib
                    safe_tag = tag.replace("::", "__").replace(" ", "_").replace("[", "").replace("]", "")
                    joblib.dump({"clf": clf, "scaler": scaler, "label_encoder": label_encoder, "classes": cls_list},
                                output_dir / f"probe_{safe_tag}.joblib")
                except Exception as exc:
                    logger.warning("Could not save probe for %s: %s", tag, exc)

            ds_k, task_k_inner, field_k = task_k.split("::", maxsplit=2)
            row_result = {
                "task_key": tag,
                "dataset_name": ds_k,
                "task_name": task_k_inner,
                "field_name": field_k + extra_tag,
                "n_train": X_tr.shape[0],
                "n_eval": X_ev.shape[0],
                "classes": cls_list,
                "metrics": metrics,
            }
            probe_results.append(row_result)
            with (output_dir / "probe_results.jsonl").open("a") as fd:
                fd.write(json.dumps(row_result) + "\n")

        _fit_and_record(X_train, X_eval, y_train, y_eval, le, e_labels_seq, task_key)

        # For tasks with a "NotHate" class (e.g. MMHS 6-class), also run a collapsed
        # binary probe: any hate type -> "Hate", NotHate stays "NotHate".
        if cfg.binary_collapse_nothate and "NotHate" in classes:
            def _to_binary(labels: list[str]) -> list[str]:
                return ["NotHate" if lbl == "NotHate" else "Hate" for lbl in labels]
            bin_train = _to_binary(t_labels_seq)
            bin_eval = _to_binary(e_labels_seq)
            le_bin = LabelEncoder()
            y_bin_train = le_bin.fit_transform(bin_train)
            y_bin_eval = le_bin.transform(bin_eval)
            _fit_and_record(X_train, X_eval, y_bin_train, y_bin_eval, le_bin, bin_eval, task_key, extra_tag=" [binary]")

        # For HarMeme-style 3-class harmfulness, collapse "somewhat harmful" + "very harmful"
        # → "harmful" vs "not harmful".
        if cfg.binary_collapse_harmful and "somewhat harmful" in classes and "very harmful" in classes:
            def _to_binary_harmful(labels: list[str]) -> list[str]:
                return ["not harmful" if lbl == "not harmful" else "harmful" for lbl in labels]
            harm_train = _to_binary_harmful(t_labels_seq)
            harm_eval = _to_binary_harmful(e_labels_seq)
            le_harm = LabelEncoder()
            y_harm_train = le_harm.fit_transform(harm_train)
            y_harm_eval = le_harm.transform(harm_eval)
            _fit_and_record(X_train, X_eval, y_harm_train, y_harm_eval, le_harm, harm_eval, task_key, extra_tag=" [harm_binary]")

    (output_dir / "top_features.json").write_text(json.dumps(top_features_all, indent=2))

    summary = {
        "config": {
            "token_selection": cfg.token_selection,
            "aggregation": cfg.aggregation,
            "eval_fraction": cfg.eval_fraction,
            "C": cfg.C,
            "class_weight": cfg.class_weight,
            "solver": cfg.solver,
        },
        "tasks": [
            {
                "task_key": r["task_key"],
                "n_train": r["n_train"],
                "n_eval": r["n_eval"],
                "accuracy": r["metrics"]["accuracy"],
                "macro_f1": r["metrics"]["macro_f1"],
                "gold_counts": r["metrics"]["gold_counts"],
            }
            for r in probe_results
        ],
    }
    (output_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Done. Results written to %s", output_dir)

    if cfg.plot:
        _plot_probe_results(probe_results, top_features_all, output_dir)


def _plot_probe_results(
    probe_results: list[dict[str, Any]],
    top_features_all: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate visualizations from probe results and top feature weights."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logging.getLogger(__name__).warning("matplotlib not available — skipping plots")
        return

    # ── 1. Macro F1 bar chart ──────────────────────────────────────────────────
    task_keys = [r["task_key"] for r in probe_results]
    macro_f1s = [r["metrics"]["macro_f1"] for r in probe_results]
    accuracies = [r["metrics"]["accuracy"] for r in probe_results]
    # Short display names
    def _short_name(tk: str) -> str:
        parts = tk.split("::")
        ds = parts[0].replace("_Dataset", "").replace("150K", "150k")
        field = parts[-1].replace("[binary]", "(bin)").replace("harmfulness", "harm").replace("label", "cls")
        return f"{ds}\n{field}"

    names = [_short_name(tk) for tk in task_keys]
    palette = plt.cm.tab10.colors
    colors = [palette[i % len(palette)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(12, max(5, len(names) * 0.6)))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, macro_f1s, height=0.5, color=colors, label="Macro F1")
    ax.barh(y_pos - 0.25, accuracies, height=0.25, color=colors, alpha=0.4, label="Accuracy")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5 baseline")
    for bar, val in zip(bars, macro_f1s):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title("SAE Probe: Macro F1 and Accuracy per Task\n(token_selection=image, max aggregation, balanced class weight)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "probe_macro_f1.png", dpi=150)
    plt.close(fig)

    # ── 2. Per-class recall heatmap ────────────────────────────────────────────
    for result in probe_results:
        pc = result["metrics"].get("per_class", {})
        if len(pc) < 2:
            continue
        labels_cls = sorted(pc.keys())
        recall_vals = np.array([[pc[lbl]["recall"] for lbl in labels_cls]])
        task_label = result["task_key"].replace("::", "\n")
        fig, ax = plt.subplots(figsize=(max(4, len(labels_cls) * 1.5), 1.8))
        im = ax.imshow(recall_vals, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(labels_cls)))
        ax.set_xticklabels(labels_cls, rotation=30, ha="right", fontsize=9)
        ax.set_yticks([])
        for j, lbl in enumerate(labels_cls):
            v = pc[lbl]
            gold = result["metrics"]["gold_counts"].get(lbl, 0)
            ax.text(j, 0, f"R={v['recall']:.2f}\nP={v['precision']:.2f}\nF1={v['f1']:.2f}\nn={gold}",
                    ha="center", va="center", fontsize=7.5, color="black")
        ax.set_title(f"Per-class Precision / Recall / F1\n{task_label}", fontsize=9)
        fig.colorbar(im, ax=ax, label="Recall", shrink=0.8)
        fig.tight_layout()
        safe = result["task_key"].replace("::", "__").replace(" ", "_").replace("[", "").replace("]", "")
        fig.savefig(output_dir / f"probe_perclass_{safe}.png", dpi=150)
        plt.close(fig)

    # ── 3. Cross-task feature overlap heatmap ─────────────────────────────────
    # Collect all unique feature indices across tasks (top-20 per class per task)
    all_feats: dict[int, dict[str, float]] = {}  # feat_idx -> {task_key: max_abs_weight}
    for tk, class_feats in top_features_all.items():
        for cls_name, feat_list in class_feats.items():
            for entry in feat_list:
                fi = entry["feature_idx"]
                w = abs(entry["weight"])
                all_feats.setdefault(fi, {})[tk] = max(all_feats.get(fi, {}).get(tk, 0), w)

    # Keep only features appearing in top-20 for 3+ task/class entries
    feat_counts = {fi: len(task_dict) for fi, task_dict in all_feats.items()}
    universal_feats = sorted(
        [fi for fi, cnt in feat_counts.items() if cnt >= 3],
        key=lambda fi: -feat_counts[fi],
    )[:40]  # top 40 most cross-task features

    if universal_feats and top_features_all:
        task_list = sorted(top_features_all.keys())
        mat = np.zeros((len(universal_feats), len(task_list)), dtype=float)
        for fi_idx, fi in enumerate(universal_feats):
            for tk_idx, tk in enumerate(task_list):
                mat[fi_idx, tk_idx] = all_feats.get(fi, {}).get(tk, 0)

        fig, ax = plt.subplots(figsize=(max(8, len(task_list) * 1.2), max(6, len(universal_feats) * 0.35)))
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(np.arange(len(task_list)))
        short_tasks = [t.split("::")[0].replace("_Dataset", "") + "\n" + t.split("::")[-1][:12] for t in task_list]
        ax.set_xticklabels(short_tasks, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(universal_feats)))
        ax.set_yticklabels([f"f{fi} (×{feat_counts[fi]})" for fi in universal_feats], fontsize=7)
        ax.set_title("Cross-task Feature Overlap (top-40 universal features)\nColor = max |weight| for that feature in that task")
        fig.colorbar(im, ax=ax, label="|weight|", shrink=0.6)
        fig.tight_layout()
        fig.savefig(output_dir / "probe_feature_overlap.png", dpi=150)
        plt.close(fig)

    logging.getLogger(__name__).info("Plots saved to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ProbeSAEConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run_probe(cfg)


if __name__ == "__main__":
    main()
