

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.residual_sae import ROLE_GENERATED, ROLE_IMAGE, ROLE_PROMPT, resolve_visible_device
from gemma3_experiments.residual_sae.matryoshka import load_matryoshka_sae
from gemma3_experiments.residual_sae.normalization import load_residual_stats


logger = logging.getLogger(__name__)


@dataclass
class MatryoshkaGranularityConfig:
    msae_path: str = ""
    train_cache_dir: str = ""
    val_cache_dir: str = ""
    output_dir: str = "./matryoshka_granularity"

    datasets: list[str] = field(default_factory=lambda: [
        "MMHS150K", "HarMeme", "Hateful_Memes_Dataset", "MAMI", "MultiOFF", "Crisis_HateMM"
    ])
    token_selection: str = "all"   # all | image | prompt | generated | image_and_generated
    aggregation: str = "max"       # max | mean
    batch_size: int = 256
    max_train_rows: int | None = None
    max_val_rows: int | None = None

    taxonomy_path: str = ""
    include_native_views: bool = True
    prefixes: list[int] = field(default_factory=list)
    min_class_samples: int = 10

    C: float = 1.0
    max_iter: int = 5000
    solver: str = "saga"
    large_dataset_solver_threshold: int = 20_000
    class_weight: str = "balanced"
    top_k_features: int = 20

    parent_child_edges: bool = True
    edge_top_parent_features: int = 50
    edge_top_child_features: int = 200
    edge_top_k_per_parent: int = 10

    device: str = "cuda:0"


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
    if token_selection == "prompt":
        return roles == ROLE_PROMPT
    if token_selection == "generated":
        return roles == ROLE_GENERATED
    if token_selection == "image_and_generated":
        return (roles == ROLE_IMAGE) | (roles == ROLE_GENERATED)
    raise ValueError(f"Unsupported token_selection: {token_selection!r}")


def _row_task_key(row: dict[str, Any]) -> str:
    primary = row["schema"]["primary_field"]
    return f"{row['dataset_name']}::{row['task_name']}::{primary['name']}"


def _row_gold_label(row: dict[str, Any]) -> str | None:
    primary = row["schema"]["primary_field"]
    value = row.get("gold_fields", {}).get(primary["name"])
    return value if isinstance(value, str) else None


def _row_sample_key(row: dict[str, Any]) -> str:
    return f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"


def _dense_acts(msae, x: torch.Tensor) -> torch.Tensor:
    acts, _ = msae.encode(x)
    return acts


def _aggregate_sample(
    msae,
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
        agg = torch.full((msae.num_latents,), -torch.inf, dtype=torch.float32, device=device)
        with torch.inference_mode():
            for start in range(0, selected.shape[0], batch_size):
                acts = _dense_acts(msae, selected[start: start + batch_size])
                agg = torch.maximum(agg, acts.max(dim=0).values.float())
        return torch.where(torch.isfinite(agg), agg, torch.zeros_like(agg)).cpu()

    if aggregation == "mean":
        total = torch.zeros((msae.num_latents,), dtype=torch.float32, device=device)
        n_tokens = 0
        with torch.inference_mode():
            for start in range(0, selected.shape[0], batch_size):
                acts = _dense_acts(msae, selected[start: start + batch_size])
                total += acts.float().sum(dim=0)
                n_tokens += acts.shape[0]
        return (total / max(n_tokens, 1)).cpu()

    raise ValueError(f"Unsupported aggregation: {aggregation!r}")


def _extract_features(
    rows: list[dict[str, Any]],
    cache_dir: Path,
    msae,
    stats,
    *,
    token_selection: str,
    aggregation: str,
    batch_size: int,
    device: str,
    n_features: int,
):
    from scipy.sparse import coo_matrix

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)

    sample_keys: list[str] = []
    kept_rows: list[dict[str, Any]] = []
    coo_rows: list[np.ndarray] = []
    coo_cols: list[np.ndarray] = []
    coo_vals: list[np.ndarray] = []
    n_extracted = 0

    for shard_idx, (shard_file, shard_rows) in enumerate(sorted(by_shard.items())):
        shard_path = cache_dir / shard_file
        assert shard_path.exists(), f"Shard missing: {shard_path}"
        data = load_file(str(shard_path))
        logger.info("Extracting %s shard %d/%d", cache_dir.name, shard_idx + 1, len(by_shard))
        for row in sorted(shard_rows, key=lambda r: int(r["offset"])):
            offset, length = int(row["offset"]), int(row["length"])
            vec = _aggregate_sample(
                msae,
                data["residual"][offset: offset + length],
                data["token_roles"][offset: offset + length],
                stats=stats,
                token_selection=token_selection,
                batch_size=batch_size,
                device=device,
                aggregation=aggregation,
            )
            if vec is None:
                continue
            nz = vec.nonzero(as_tuple=True)[0]
            if len(nz):
                coo_rows.append(np.full(len(nz), n_extracted, dtype=np.int32))
                coo_cols.append(nz.to(torch.int32).numpy())
                coo_vals.append(vec[nz].to(torch.float32).numpy())
            sample_keys.append(_row_sample_key(row))
            kept_rows.append(row)
            n_extracted += 1

    rows_arr = np.concatenate(coo_rows) if coo_rows else np.array([], dtype=np.int32)
    cols_arr = np.concatenate(coo_cols) if coo_cols else np.array([], dtype=np.int32)
    vals_arr = np.concatenate(coo_vals) if coo_vals else np.array([], dtype=np.float32)
    csr = coo_matrix((vals_arr, (rows_arr, cols_arr)), shape=(n_extracted, n_features), dtype=np.float32).tocsr()
    return sample_keys, kept_rows, csr


def _load_taxonomy(path: str) -> list[dict[str, Any]]:
    if not path:
        return []
    with Path(path).open() as fd:
        raw = json.load(fd)
    return list(raw.get("views", []))


def _taxonomy_label(row: dict[str, Any], view: dict[str, Any]) -> str | None:
    for rule in view.get("rules", []):
        if rule.get("dataset") and row.get("dataset_name") != rule["dataset"]:
            continue
        if rule.get("task") and row.get("task_name") != rule["task"]:
            continue
        field_name = rule.get("field") or row["schema"]["primary_field"]["name"]
        raw = row.get("gold_fields", {}).get(field_name)
        mapped = rule.get("map", {}).get(raw)
        if mapped is not None:
            return mapped
    return None


def _per_class_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    per_class = {}
    macro_f1 = 0.0
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(y_true, y_pred))
        fp = sum(g != label and p == label for g, p in zip(y_true, y_pred))
        fn = sum(g == label and p != label for g, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        macro_f1 += f1
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    correct = sum(g == p for g, p in zip(y_true, y_pred))
    return {
        "accuracy": correct / len(y_true) if y_true else 0.0,
        "macro_f1": macro_f1 / len(labels) if labels else 0.0,
        "per_class": per_class,
        "gold_counts": dict(Counter(y_true)),
        "pred_counts": dict(Counter(y_pred)),
    }


def _fit_probe(X_train, y_train_raw, X_val, y_val_raw, cfg: MatryoshkaGranularityConfig):
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import LabelEncoder, MaxAbsScaler

    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    effective_solver = "liblinear" if len(y_train_raw) > cfg.large_dataset_solver_threshold else cfg.solver
    base = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        solver=effective_solver,
        class_weight=cfg.class_weight if cfg.class_weight != "None" else None,
    )
    clf = OneVsRestClassifier(base) if effective_solver == "liblinear" and len(le.classes_) >= 3 else base
    clf.fit(X_train, y_train)
    pred = le.inverse_transform(clf.predict(X_val)).tolist()
    return clf, le, _per_class_metrics(y_val_raw, pred)


def _iter_label_views(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    cfg: MatryoshkaGranularityConfig,
):
    views: list[tuple[str, dict[str, str], dict[str, str]]] = []
    if cfg.include_native_views:
        task_keys = sorted({_row_task_key(r) for r in train_rows} | {_row_task_key(r) for r in val_rows})
        for task_key in task_keys:
            tr = {_row_sample_key(r): _row_gold_label(r) for r in train_rows if _row_task_key(r) == task_key}
            va = {_row_sample_key(r): _row_gold_label(r) for r in val_rows if _row_task_key(r) == task_key}
            views.append((f"native::{task_key}", tr, va))

    for view in _load_taxonomy(cfg.taxonomy_path):
        tr = {_row_sample_key(r): _taxonomy_label(r, view) for r in train_rows}
        va = {_row_sample_key(r): _taxonomy_label(r, view) for r in val_rows}
        views.append((f"taxonomy::{view['name']}", tr, va))

    for name, tr_map, va_map in views:
        tr_map = {k: v for k, v in tr_map.items() if isinstance(v, str)}
        va_map = {k: v for k, v in va_map.items() if isinstance(v, str)}
        if tr_map and va_map:
            yield name, tr_map, va_map


def _write_parent_child_edges(train_csr, prefixes: list[int], output_path: Path, cfg: MatryoshkaGranularityConfig) -> None:
    if len(prefixes) < 2:
        return
    with output_path.open("w") as fd:
        for parent_k, child_k in zip(prefixes, prefixes[1:]):
            parent = train_csr[:, :parent_k]
            child = train_csr[:, parent_k:child_k]
            if parent.shape[1] == 0 or child.shape[1] == 0:
                continue
            parent_activity = np.asarray(parent.sum(axis=0)).ravel()
            child_activity = np.asarray(child.sum(axis=0)).ravel()
            parent_cols = parent_activity.argsort()[::-1][: cfg.edge_top_parent_features]
            child_cols_local = child_activity.argsort()[::-1][: cfg.edge_top_child_features]
            parent_sub = parent[:, parent_cols].astype(np.float32)
            child_sub = child[:, child_cols_local].astype(np.float32)
            scores = (parent_sub.T @ child_sub).toarray()
            parent_norm = np.sqrt(np.asarray(parent_sub.power(2).sum(axis=0)).ravel()).clip(min=1e-8)
            child_norm = np.sqrt(np.asarray(child_sub.power(2).sum(axis=0)).ravel()).clip(min=1e-8)
            scores = scores / parent_norm[:, None] / child_norm[None, :]
            for i, parent_feature in enumerate(parent_cols):
                top = scores[i].argsort()[::-1][: cfg.edge_top_k_per_parent]
                for j in top:
                    fd.write(json.dumps({
                        "parent_prefix": parent_k,
                        "child_prefix": child_k,
                        "parent_feature": int(parent_feature),
                        "child_feature": int(parent_k + child_cols_local[j]),
                        "cosine_coactivation": float(scores[i, j]),
                    }) + "\n")


def analyze_matryoshka_granularity(cfg: MatryoshkaGranularityConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_visible_device(cfg.device)
    msae = load_matryoshka_sae(cfg.msae_path, device=device)
    stats = load_residual_stats(cfg.msae_path)
    prefixes = sorted(set(cfg.prefixes or msae.prefix_sizes))
    prefixes = [k for k in prefixes if 0 < k <= msae.num_latents]
    if msae.num_latents not in prefixes:
        prefixes.append(msae.num_latents)
    logger.info("Analyzing MSAE prefixes: %s", prefixes)

    datasets = set(cfg.datasets)
    train_cache = Path(cfg.train_cache_dir)
    train_rows = [
        r for r in _read_jsonl(train_cache / "manifest.jsonl")
        if r.get("dataset_name") in datasets and r.get("split") == "train" and _row_gold_label(r) is not None
    ]
    if cfg.max_train_rows is not None:
        train_rows = train_rows[: cfg.max_train_rows]
    train_keys, train_kept_rows, train_csr = _extract_features(
        train_rows, train_cache, msae, stats,
        token_selection=cfg.token_selection,
        aggregation=cfg.aggregation,
        batch_size=cfg.batch_size,
        device=device,
        n_features=msae.num_latents,
    )
    train_key_to_idx = {k: i for i, k in enumerate(train_keys)}

    if cfg.val_cache_dir:
        val_cache = Path(cfg.val_cache_dir)
        val_rows = [
            r for r in _read_jsonl(val_cache / "manifest.jsonl")
            if r.get("dataset_name") in datasets and r.get("split") in {"validation", "val", "dev"} and _row_gold_label(r) is not None
        ]
        if cfg.max_val_rows is not None:
            val_rows = val_rows[: cfg.max_val_rows]
        val_keys, val_kept_rows, val_csr = _extract_features(
            val_rows, val_cache, msae, stats,
            token_selection=cfg.token_selection,
            aggregation=cfg.aggregation,
            batch_size=cfg.batch_size,
            device=device,
            n_features=msae.num_latents,
        )
        val_key_to_idx = {k: i for i, k in enumerate(val_keys)}
    else:
        # Analysis fallback: use train rows as validation too. This is only for
        # feature-granularity inspection when a validation cache is unavailable.
        val_kept_rows = train_kept_rows
        val_csr = train_csr
        val_key_to_idx = train_key_to_idx

    metrics_path = output_dir / "prefix_probe_metrics.jsonl"
    top_features_path = output_dir / "prefix_top_features.jsonl"
    with metrics_path.open("w") as metrics_fd, top_features_path.open("w") as top_fd:
        for view_name, tr_labels_by_key, va_labels_by_key in _iter_label_views(train_kept_rows, val_kept_rows, cfg):
            common_train = [k for k in train_keys if k in tr_labels_by_key]
            common_val = [k for k in val_key_to_idx if k in va_labels_by_key]
            if not common_train or not common_val:
                continue
            y_train = [tr_labels_by_key[k] for k in common_train]
            counts = Counter(y_train)
            if len(counts) < 2 or any(c < cfg.min_class_samples for c in counts.values()):
                continue
            train_classes = set(counts)
            common_val = [k for k in common_val if va_labels_by_key[k] in train_classes]
            y_val = [va_labels_by_key[k] for k in common_val]
            if not common_val:
                continue
            train_idx = [train_key_to_idx[k] for k in common_train]
            val_idx = [val_key_to_idx[k] for k in common_val]

            for prefix in prefixes:
                X_train = train_csr[train_idx, :prefix]
                X_val = val_csr[val_idx, :prefix]
                clf, le, metrics = _fit_probe(X_train, y_train, X_val, y_val, cfg)
                row = {
                    "view": view_name,
                    "prefix": prefix,
                    "n_train": len(y_train),
                    "n_val": len(y_val),
                    "classes": le.classes_.tolist(),
                    **metrics,
                }
                metrics_fd.write(json.dumps(row) + "\n")
                logger.info(
                    "%s prefix=%d n_train=%d n_val=%d macro_f1=%.4f acc=%.4f",
                    view_name, prefix, len(y_train), len(y_val), metrics["macro_f1"], metrics["accuracy"],
                )

                estimators = getattr(clf, "estimators_", None)
                if estimators is not None:
                    coef_by_class = [(str(cls), est.coef_.ravel()) for cls, est in zip(le.classes_, estimators)]
                elif len(le.classes_) == 2:
                    coef_by_class = [(str(le.classes_[1]), clf.coef_.ravel())]
                else:
                    coef_by_class = [(str(cls), coef) for cls, coef in zip(le.classes_, clf.coef_)]
                for class_name, weights in coef_by_class:
                    top_idx = np.abs(weights).argsort()[::-1][: cfg.top_k_features]
                    for rank, feat in enumerate(top_idx, start=1):
                        top_fd.write(json.dumps({
                            "view": view_name,
                            "prefix": prefix,
                            "class": class_name,
                            "rank": rank,
                            "feature_idx": int(feat),
                            "weight": float(weights[feat]),
                            "abs_weight": float(abs(weights[feat])),
                        }) + "\n")

    if cfg.parent_child_edges:
        _write_parent_child_edges(train_csr, prefixes, output_dir / "parent_child_edges.jsonl", cfg)

    with (output_dir / "run_config.json").open("w") as fd:
        json.dump({**cfg.__dict__, "resolved_prefixes": prefixes}, fd, indent=2)


def main() -> None:
    cfg = parse(MatryoshkaGranularityConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    analyze_matryoshka_granularity(cfg)


if __name__ == "__main__":
    main()
