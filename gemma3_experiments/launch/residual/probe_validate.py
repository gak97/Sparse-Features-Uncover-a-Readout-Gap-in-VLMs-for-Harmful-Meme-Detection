

import json
import logging
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.residual_sae import (
    ROLE_GENERATED,
    ROLE_IMAGE,
    ROLE_PROMPT,
    load_residual_sae,
    load_residual_stats,
    resolve_visible_device,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────


@dataclass
class ProbeValidateConfig:
    residual_sae_path: str = ""
    train_cache_dir: str = ""
    # Primary validation cache dir (required). Additional dirs merged in.
    val_cache_dir: str = ""
    eval_splits: list[str] = field(default_factory=lambda: ["validation", "val", "dev"])
    # Additional validation cache dirs (e.g. residual_validation_crisis). Merged with val_cache_dir.
    extra_val_cache_dirs: list[str] = field(default_factory=list)
    baseline_predictions_path: str = ""
    output_dir: str = ""

    datasets: list[str] = field(default_factory=lambda: [
        "MMHS150K", "HarMeme", "Hateful_Memes_Dataset", "MAMI", "MultiOFF", "Crisis_HateMM"
    ])

    token_selection: str = "image"   # image | all | generated | image_and_generated
    aggregation: str = "max"         # max | mean
    batch_size: int = 256

    min_class_samples: int = 10

    # Run a two-stage hierarchical probe for MMHS: binary (NotHate/Hate) then hate-type.
    hierarchical_mmhs: bool = True
    # Analyse FHM predictions per sample to flag potential benign confounders.
    fhm_confounder_analysis: bool = True

    C: float = 1.0
    max_iter: int = 5000
    solver: str = "saga"
    class_weight: str = "balanced"
    large_dataset_solver_threshold: int = 20_000

    # Per-dataset C overrides (key = dataset_name prefix, e.g. "Hateful_Memes_Dataset").
    # When non-empty, the dataset's probe is fitted with this C instead of cfg.C.
    # Useful for applying stronger regularisation to noisy datasets while leaving others
    # at the default. Example: {"Hateful_Memes_Dataset": 0.1}
    per_task_C_overrides: dict[str, float] = field(default_factory=dict)

    binary_collapse_nothate: bool = True
    # For HarMeme-style 3-class harmfulness tasks, also run a binary collapse:
    # "somewhat harmful" + "very harmful" → "harmful" vs "not harmful".
    binary_collapse_harmful: bool = True
    top_k_features: int = 20

    model_agree_train_filter_datasets: list[str] = field(
        default_factory=lambda: ["Hateful_Memes_Dataset"]
    )

    concat_token_selections: list[str] = field(default_factory=list)
    # Optional explicit interaction blocks to append to concat features. Each entry is
    # "left:right", e.g. "image:prompt" adds abs(image_features - prompt_features).
    # This tests whether FHM needs direct cross-modal mismatch features beyond MLP
    # hidden-layer interactions over pooled all-token features.
    interaction_token_selection_pairs: list[str] = field(default_factory=list)

    # For listed dataset names, after fitting the probe sweep the binary classification
    # threshold [0.10, 0.90] on the validation set to find the threshold maximising
    # macro-F1.  Reports "calibrated" metrics alongside the standard 0.5-threshold
    # result.  Only applies to binary tasks (2 classes). This uses the val set for
    # threshold selection so scores are optimistic — treat as an upper-bound diagnostic.
    calibrate_threshold_datasets: list[str] = field(
        default_factory=lambda: ["Hateful_Memes_Dataset"]
    )

    classifier_type: str = "logistic"   # logistic | mlp
    mlp_hidden_size: int = 512
    mlp_dropout: float = 0.3
    mlp_max_iter: int = 200
    # MLP alpha (L2 regularisation) — higher values reduce overfitting on small datasets.
    mlp_alpha: float = 1e-4

    hard_negative_mining_datasets: list[str] = field(default_factory=list)
    # Optional FHM metadata file with pseudo confounder IDs.  When set, hard-negative
    # mining uses pseudo_img_idx / pseudo_text_idx groups instead of only exact text.
    fhm_confounder_metadata_path: str = ""
    # Confounder sources used when fhm_confounder_metadata_path is set:
    #   pseudo_img  = same background image, different text
    #   pseudo_text = same meme text, different image
    fhm_confounder_pair_sources: list[str] = field(default_factory=lambda: ["pseudo_img", "pseudo_text"])

    plot: bool = True
    # Save validation feature matrix to <output_dir>/feature_matrix.npz for
    # downstream causal ablation and other post-hoc analyses.
    save_feature_matrix: bool = True
    save_probe_for_datasets: list[str] = field(default_factory=list)
    # Fit one-vs-rest binary probes per MMHS hate category (generated tokens only).
    # For each of the 6 classes, fits a separate binary LR with balanced weights,
    # sweeps threshold on validation, and reports per-class AUC + calibrated F1/P/R.
    # Useful for class-specific interpretability where the joint 6-class softmax
    # is sub-optimal for minority classes (Sexist, Religion).
    mmhs_ovr: bool = False
    calibration_fraction: float = 0.0

    # ── Probe validity controls ──────────────────────────────────────────────
    # Shuffled-label control: re-fit each probe with randomly permuted training
    # labels.  A genuine probe should drop to chance accuracy (~1/n_classes).
    # Stored as task_result["shuffled_label_control"] in probe_val_metrics.json.
    shuffled_label_control: bool = False
    # Column-permutation control: randomly permute the feature-column index
    # assignments before fitting (same sparsity pattern, SAE semantics destroyed).
    # If the permuted probe matches the real probe, the specific SAE latents do
    # not matter — only the aggregate activation statistics do.
    column_permutation_control: bool = False
    # Seed for both controls (reproducible permutations).
    control_seed: int = 0
    # Random-sparse-features control: replace SAE features with random sparse
    # vectors of the same shape, per-sample L0, and value range.  If the random
    # probe matches the real probe, the SAE's specific sparse encoding structure
    # is not informative — only having ~L0 active dimensions per sample matters.
    random_features_control: bool = False
    # Raw-residual-stream control: fit the same probe on the 2560-dim hidden
    # states from the dense cache (before SAE encoding).  If raw-residual ≈ SAE,
    # the SAE's sparse factorisation adds no discriminative value over the dense
    # representation.  Train and val rows come from different dense caches so two
    # dirs must be provided; leave either empty to skip the control.
    raw_residual_control_train_cache_dir: str = ""
    raw_residual_control_val_cache_dir: str = ""

    device: str = "cuda:0"


def _write_feature_matrix(path: Path, keys: list[str], X: Any) -> None:
    np.savez_compressed(path, keys=np.array(keys, dtype=object), X=X)


def _make_random_sparse_like(csr: Any, rng: "np.random.Generator") -> Any:
    """Random sparse matrix with same shape, per-row L0, and value range.

    Used as 'random dictionary' null control: destroys SAE-specific structure
    while preserving sparsity and activation magnitude statistics.
    """
    from scipy.sparse import csr_matrix as _csr
    n_samples, n_features = csr.shape
    l0_per_row = np.diff(csr.indptr)
    total_nnz = int(l0_per_row.sum())
    val_min = float(csr.data.min()) if csr.nnz > 0 else 0.0
    val_max = float(csr.data.max()) if csr.nnz > 0 else 1.0
    rand_cols = rng.integers(0, n_features, size=total_nnz)
    rand_vals = rng.uniform(val_min, val_max, size=total_nnz).astype(np.float32)
    row_idx   = np.repeat(np.arange(n_samples, dtype=np.int32), l0_per_row)
    return _csr((rand_vals, (row_idx, rand_cols)), shape=(n_samples, n_features), dtype=np.float32)


def _extract_raw_features(
    rows: list[dict[str, Any]],
    dense_cache_dir: Path,
    *,
    token_selection: str,
    aggregation: str,
    hidden_dim: int = 2560,
) -> tuple[list[str], "np.ndarray"]:
    """Extract raw hidden-state features from the dense cache (no SAE encoding).

    Reads the `hidden` tensor from dense shards instead of `residual` from SAE
    shards.  Returns (sample_keys, X) where X is a float32 array of shape
    (n_samples, hidden_dim).  Used for the raw-residual-stream control.
    """
    from collections import defaultdict as _dd
    import torch as _torch
    from safetensors.torch import load_file as _lf

    by_shard: dict[str, list[dict[str, Any]]] = _dd(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)

    # Map residual shard names → dense shard names  (e.g. residual_w00_00000 → dense_w00_00000)
    def _to_dense_fname(fname: str) -> str:
        return fname.replace("residual_", "dense_", 1)

    sample_keys: list[str] = []
    vecs: list[np.ndarray] = []
    skipped = 0

    for shard_file, shard_rows in sorted(by_shard.items()):
        dense_shard = dense_cache_dir / _to_dense_fname(shard_file)
        if not dense_shard.exists():
            skipped += len(shard_rows)
            continue
        shard_data = _lf(str(dense_shard))
        hidden    = shard_data["hidden"]        # [T, hidden_dim]
        token_roles = shard_data["token_roles"]  # [T]
        for row in sorted(shard_rows, key=lambda r: int(r["offset"])):
            offset, length = int(row["offset"]), int(row["length"])
            h  = hidden[offset: offset + length]         # [L, hidden_dim]
            tr = token_roles[offset: offset + length]    # [L]
            mask = _role_mask(tr, token_selection)
            if not bool(mask.any()):
                skipped += 1
                continue
            sel = h[mask].float()  # [n_sel, hidden_dim]
            if aggregation == "max":
                vec = sel.max(dim=0).values
            elif aggregation == "mean":
                vec = sel.mean(dim=0)
            else:
                raise ValueError(f"Unsupported aggregation: {aggregation!r}")
            sample_keys.append(_row_sample_key(row))
            vecs.append(vec.cpu().numpy())

    if not vecs:
        return [], np.zeros((0, hidden_dim), dtype=np.float32)
    return sample_keys, np.stack(vecs, axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Shared utilities (mirror of probe_sae_features.py)
# ──────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_model_generated_label(row: dict[str, Any]) -> str | None:
    """Extract the model's own predicted label from the manifest's generated_text field.

    The generated_text is a JSON blob produced by Gemma3 during the dense caching pass.
    This is used to filter training rows where the model's reasoning contradicts gold —
    those rows have contaminated generated-token SAE features.
    """
    import re
    gt = row.get("generated_text", "") or ""
    m = re.search(r'"label"\s*:\s*"([^"]+)"', gt)
    return m.group(1) if m else None


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


def _row_sample_key(row: dict[str, Any]) -> str:
    return f"{row['dataset_name']}::{row['task_name']}::{row['sample_id']}"


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
        per_class[lbl] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                          "tp": tp, "fp": fp, "fn": fn}
        macro_p += p
        macro_r += r
        macro_f1 += f
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
# Feature extraction (shared logic for train and val caches)
# ──────────────────────────────────────────────────────────────


def _extract_features(
    rows: list[dict[str, Any]],
    cache_dir: Path,
    sae,
    stats,
    *,
    token_selection: str,
    aggregation: str,
    batch_size: int,
    device: str,
    n_features: int,
):
    pass
    from scipy.sparse import coo_matrix

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[row["shard_file"]].append(row)

    sample_keys: list[str] = []
    coo_rows: list[np.ndarray] = []
    coo_cols: list[np.ndarray] = []
    coo_vals: list[np.ndarray] = []
    skipped = 0
    n_extracted = 0

    for shard_idx, (shard_file, shard_rows) in enumerate(sorted(by_shard.items())):
        shard_path = cache_dir / shard_file
        assert shard_path.exists(), f"Shard missing: {shard_path}"
        shard_data = load_file(str(shard_path))

        if (shard_idx + 1) % 100 == 0 or shard_idx == 0:
            logger.info(
                "Extracting features shard %d/%d (%d rows, %d extracted so far)",
                shard_idx + 1, len(by_shard), len(shard_rows), n_extracted,
            )

        for row in sorted(shard_rows, key=lambda r: int(r["offset"])):
            offset, length = int(row["offset"]), int(row["length"])
            residual = shard_data["residual"][offset: offset + length]
            roles = shard_data["token_roles"][offset: offset + length]

            vec = _aggregate_sample(
                sae, residual, roles,
                stats=stats,
                token_selection=token_selection,
                batch_size=batch_size,
                device=device,
                aggregation=aggregation,
            )
            if vec is None:
                skipped += 1
                continue

            # Accumulate as sparse triplets — never store full dense 20k-dim array.
            nz = vec.nonzero(as_tuple=True)[0]
            if len(nz):
                coo_rows.append(np.full(len(nz), n_extracted, dtype=np.int32))
                coo_cols.append(nz.to(torch.int32).numpy())
                coo_vals.append(vec[nz].to(torch.float32).numpy())

            sample_keys.append(_row_sample_key(row))
            n_extracted += 1

    logger.info(
        "Feature extraction complete: %d extracted, %d skipped (no tokens matched %s)",
        n_extracted, skipped, token_selection,
    )

    if coo_rows:
        rows_arr = np.concatenate(coo_rows)
        cols_arr = np.concatenate(coo_cols)
        vals_arr = np.concatenate(coo_vals)
    else:
        rows_arr = cols_arr = np.array([], dtype=np.int32)
        vals_arr = np.array([], dtype=np.float32)

    csr = coo_matrix(
        (vals_arr, (rows_arr, cols_arr)),
        shape=(n_extracted, n_features),
        dtype=np.float32,
    ).tocsr()
    return sample_keys, csr


def _extract_features_multi(
    rows: list[dict[str, Any]],
    sae,
    stats,
    *,
    token_selection: str,
    aggregation: str,
    batch_size: int,
    device: str,
    n_features: int,
):
    pass
    from scipy.sparse import coo_matrix

    # Group by (cache_dir, shard_file) — NOT just shard_file, to avoid cross-cache collisions.
    by_shard: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cache_dir_str = row["_cache_dir"]
        by_shard[(cache_dir_str, row["shard_file"])].append(row)

    sample_keys: list[str] = []
    coo_rows_list: list[np.ndarray] = []
    coo_cols_list: list[np.ndarray] = []
    coo_vals_list: list[np.ndarray] = []
    skipped = 0
    n_extracted = 0

    for shard_idx, ((cache_dir_str, shard_file), shard_rows) in enumerate(sorted(by_shard.items())):
        shard_path = Path(cache_dir_str) / shard_file
        assert shard_path.exists(), f"Shard missing: {shard_path}"
        shard_data = load_file(str(shard_path))

        if (shard_idx + 1) % 100 == 0 or shard_idx == 0:
            logger.info(
                "Extracting features shard %d/%d (%d rows, %d extracted so far)",
                shard_idx + 1, len(by_shard), len(shard_rows), n_extracted,
            )

        for row in sorted(shard_rows, key=lambda r: int(r["offset"])):
            offset, length = int(row["offset"]), int(row["length"])
            residual = shard_data["residual"][offset: offset + length]
            roles = shard_data["token_roles"][offset: offset + length]

            vec = _aggregate_sample(
                sae, residual, roles,
                stats=stats,
                token_selection=token_selection,
                batch_size=batch_size,
                device=device,
                aggregation=aggregation,
            )
            if vec is None:
                skipped += 1
                continue

            nz = vec.nonzero(as_tuple=True)[0]
            if len(nz):
                coo_rows_list.append(np.full(len(nz), n_extracted, dtype=np.int32))
                coo_cols_list.append(nz.to(torch.int32).numpy())
                coo_vals_list.append(vec[nz].to(torch.float32).numpy())

            sample_keys.append(_row_sample_key(row))
            n_extracted += 1

    logger.info(
        "Feature extraction complete: %d extracted, %d skipped (no tokens matched %s)",
        n_extracted, skipped, token_selection,
    )

    if coo_rows_list:
        rows_arr = np.concatenate(coo_rows_list)
        cols_arr = np.concatenate(coo_cols_list)
        vals_arr = np.concatenate(coo_vals_list)
    else:
        rows_arr = cols_arr = np.array([], dtype=np.int32)
        vals_arr = np.array([], dtype=np.float32)

    csr = coo_matrix(
        (vals_arr, (rows_arr, cols_arr)),
        shape=(n_extracted, n_features),
        dtype=np.float32,
    ).tocsr()
    return sample_keys, csr


def _align_and_concat_csrs(
    key_csr_pairs: list[tuple[list[str], Any]],
) -> tuple[list[str], Any]:
    pass
    from scipy.sparse import hstack

    assert len(key_csr_pairs) >= 2, "Need at least two feature sets to concatenate"

    # Common sample keys — preserve ordering from the first extraction.
    all_sets = [set(keys) for keys, _ in key_csr_pairs]
    common_set: set[str] = all_sets[0].intersection(*all_sets[1:])
    ref_keys = key_csr_pairs[0][0]
    common_keys = [k for k in ref_keys if k in common_set]

    aligned: list[Any] = []
    for keys, csr in key_csr_pairs:
        key_to_idx = {k: i for i, k in enumerate(keys)}
        select = [key_to_idx[k] for k in common_keys]
        aligned.append(csr[select])

    return common_keys, hstack(aligned, format="csr")


def _parse_interaction_pair(pair: str) -> tuple[str, str]:
    for sep in (":", ",", "+"):
        if sep in pair:
            left, right = [part.strip() for part in pair.split(sep, maxsplit=1)]
            if left and right:
                return left, right
    raise ValueError(
        f"Unsupported interaction pair {pair!r}; expected 'left:right', e.g. 'image:prompt'."
    )


def _required_token_selections(
    concat_token_selections: list[str],
    interaction_pairs: list[str],
) -> list[str]:
    required = list(concat_token_selections)
    for pair in interaction_pairs:
        left, right = _parse_interaction_pair(pair)
        required.extend([left, right])
    return list(dict.fromkeys(required))


def _align_csr(keys: list[str], csr: Any, common_keys: list[str]) -> Any:
    key_to_idx = {key: i for i, key in enumerate(keys)}
    select = [key_to_idx[key] for key in common_keys]
    return csr[select]


def _absdiff_csr(left: Any, right: Any) -> Any:
    diff = (left - right).tocsr()
    diff.data = np.abs(diff.data)
    diff.eliminate_zeros()
    return diff


def _build_feature_blocks(
    feature_sets: dict[str, tuple[list[str], Any]],
    concat_token_selections: list[str],
    interaction_pairs: list[str],
) -> tuple[list[str], Any]:
    from scipy.sparse import hstack

    required = _required_token_selections(concat_token_selections, interaction_pairs)
    assert required, "Expected at least one concat selection or interaction pair."
    missing = sorted(set(required) - set(feature_sets))
    assert not missing, f"Missing extracted feature sets: {missing}"

    common_set = set(feature_sets[required[0]][0])
    for token_selection in required[1:]:
        common_set &= set(feature_sets[token_selection][0])
    common_keys = [key for key in feature_sets[required[0]][0] if key in common_set]
    assert common_keys, "No common sample keys across requested feature blocks."

    aligned_by_selection = {
        token_selection: _align_csr(keys, csr, common_keys)
        for token_selection, (keys, csr) in feature_sets.items()
        if token_selection in required
    }
    blocks = [aligned_by_selection[token_selection] for token_selection in concat_token_selections]
    for pair in interaction_pairs:
        left, right = _parse_interaction_pair(pair)
        blocks.append(_absdiff_csr(aligned_by_selection[left], aligned_by_selection[right]))
    assert blocks, "No feature blocks built."
    return common_keys, hstack(blocks, format="csr")


def _calibrate_threshold(
    y_true: list[str],
    y_proba: Any,          # numpy array (n_val, n_classes)
    classes: list[str],    # le.classes_ order
    pos_class: str,        # class to treat as positive (e.g. "hateful")
) -> tuple[float, float, list[str]]:
    """Sweep classification threshold for pos_class to maximise macro-F1.

    Returns (best_threshold, best_macro_f1, best_pred_list).
    Only meaningful for binary tasks; for multi-class `pos_class` is used as the
    positive class in a one-vs-rest threshold sweep.
    """
    if pos_class not in classes:
        return 0.5, 0.0, []
    pos_idx = classes.index(pos_class)
    best_t, best_f1, best_preds = 0.5, 0.0, []
    for t_int in range(10, 91):
        t = t_int / 100.0
        pred_enc = (y_proba[:, pos_idx] >= t).astype(int)
        # For binary: 0 = neg class, 1 = pos class based on encoding direction
        # classes is sorted, so we need to know if pos_class is index 0 or 1.
        preds = [classes[pos_idx] if e == 1 else classes[1 - pos_idx] for e in pred_enc]
        m = _per_class_metrics(y_true, preds)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_t = t
            best_preds = preds
    return best_t, best_f1, best_preds


def _build_hard_negative_pairs(
    train_rows: list[dict[str, Any]],
    train_csr: Any,
    train_key_to_idx: dict[str, int],
    dataset_name: str,
    text_field: str = "meme_text",
) -> tuple[Any, list[str], dict[str, Any]]:
    pass
    import scipy.sparse

    # Group rows by lowercased meme text
    by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in train_rows:
        if r.get("dataset_name") != dataset_name:
            continue
        text = (r.get(text_field) or "").lower().strip()
        if text:
            by_text[text].append(r)

    delta_rows_list = []
    delta_labels = []
    n_pairs = 0
    for text, rows in by_text.items():
        # Need at least one hateful and one not hateful
        hateful_rows = [r for r in rows if _row_gold_label(r) in ("hateful", "Hateful", "1", "yes")]
        benign_rows  = [r for r in rows if _row_gold_label(r) in ("not hateful", "Not Hateful", "0", "no")]
        if not hateful_rows or not benign_rows:
            continue
        for hr in hateful_rows:
            hk = _row_sample_key(hr)
            if hk not in train_key_to_idx:
                continue
            for br in benign_rows:
                bk = _row_sample_key(br)
                if bk not in train_key_to_idx:
                    continue
                h_vec = train_csr[train_key_to_idx[hk]]
                b_vec = train_csr[train_key_to_idx[bk]]
                # hateful - benign → hateful synthetic sample
                delta_rows_list.append(h_vec - b_vec)
                delta_labels.append(_row_gold_label(hr))
                # benign - hateful → benign synthetic sample
                delta_rows_list.append(b_vec - h_vec)
                delta_labels.append(_row_gold_label(br))
                n_pairs += 1

    if not delta_rows_list:
        logger.info("hard_negative_mining: no text-confounder pairs found for %s", dataset_name)
        return scipy.sparse.csr_matrix((0, train_csr.shape[1]), dtype=np.float32), [], {
            "mode": "manifest_text",
            "dataset_name": dataset_name,
            "n_pairs": 0,
            "n_synthetic_rows": 0,
        }

    delta_csr = scipy.sparse.vstack(delta_rows_list, format="csr").astype(np.float32)
    logger.info(
        "hard_negative_mining: built %d synthetic delta rows from %d pairs for %s",
        len(delta_labels), n_pairs, dataset_name,
    )
    return delta_csr, delta_labels, {
        "mode": "manifest_text",
        "dataset_name": dataset_name,
        "n_pairs": n_pairs,
        "n_synthetic_rows": len(delta_labels),
    }


def _normalise_group_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text if text else None


def _normalise_fhm_id(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _load_fhm_confounder_metadata(path: Path) -> dict[str, dict[str, Any]]:
    assert path.exists(), f"FHM confounder metadata not found: {path}"
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        rows = list(data.values())
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported FHM confounder metadata format: {path}")

    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_id = row.get("id")
        if sample_id is None and row.get("img"):
            sample_id = Path(str(row["img"])).stem
        if sample_id is None:
            continue
        by_id[_normalise_fhm_id(sample_id)] = row
    return by_id


def _build_fhm_metadata_hard_negative_pairs(
    train_rows: list[dict[str, Any]],
    train_csr: Any,
    train_key_to_idx: dict[str, int],
    metadata_path: Path,
    pair_sources: list[str],
) -> tuple[Any, list[str], dict[str, Any]]:
    """Build FHM hard negatives from pseudo image/text confounder IDs.

    `pseudo_img_idx` groups share a background image with different text; deltas
    isolate text/context flips. `pseudo_text_idx` groups share meme text with
    different images; deltas isolate image/context flips.
    """
    import scipy.sparse

    source_to_col = {
        "pseudo_img": "pseudo_img_idx",
        "pseudo_image": "pseudo_img_idx",
        "pseudo_text": "pseudo_text_idx",
    }
    unknown = sorted(set(pair_sources) - set(source_to_col))
    if unknown:
        raise ValueError(f"Unsupported fhm_confounder_pair_sources: {unknown}")

    metadata = _load_fhm_confounder_metadata(metadata_path)
    rows_by_key = {
        _row_sample_key(row): row
        for row in train_rows
        if row.get("dataset_name") == "Hateful_Memes_Dataset"
        and _row_sample_key(row) in train_key_to_idx
    }
    groups_by_source: dict[str, dict[str, list[dict[str, Any]]]] = {
        source: defaultdict(list) for source in pair_sources
    }
    n_rows_with_metadata = 0

    for row in rows_by_key.values():
        meta = metadata.get(_normalise_fhm_id(row.get("sample_id")))
        if meta is None:
            continue
        n_rows_with_metadata += 1
        for source in pair_sources:
            group_value = _normalise_group_value(meta.get(source_to_col[source]))
            if group_value is not None:
                groups_by_source[source][group_value].append(row)

    delta_rows_list = []
    delta_labels: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    source_stats: dict[str, dict[str, int]] = {}

    for source, groups in groups_by_source.items():
        n_groups = 0
        n_pairs = 0
        for rows in groups.values():
            labels = {_row_gold_label(row) for row in rows}
            if len(rows) < 2 or len(labels) < 2:
                continue
            hateful_rows = [row for row in rows if _row_gold_label(row) in ("hateful", "Hateful", "1", "yes")]
            benign_rows = [row for row in rows if _row_gold_label(row) in ("not hateful", "Not Hateful", "0", "no")]
            if not hateful_rows or not benign_rows:
                continue
            n_groups += 1
            for hr in hateful_rows:
                hk = _row_sample_key(hr)
                for br in benign_rows:
                    bk = _row_sample_key(br)
                    pair_key = tuple(sorted((hk, bk)))
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    h_vec = train_csr[train_key_to_idx[hk]]
                    b_vec = train_csr[train_key_to_idx[bk]]
                    delta_rows_list.append(h_vec - b_vec)
                    delta_labels.append(_row_gold_label(hr))
                    delta_rows_list.append(b_vec - h_vec)
                    delta_labels.append(_row_gold_label(br))
                    n_pairs += 1
        source_stats[source] = {"n_groups": n_groups, "n_pairs": n_pairs}

    stats = {
        "mode": "fhm_metadata",
        "metadata_path": str(metadata_path),
        "pair_sources": pair_sources,
        "n_metadata_rows": len(metadata),
        "n_train_rows_with_metadata": n_rows_with_metadata,
        "n_pairs": len(seen_pairs),
        "n_synthetic_rows": len(delta_labels),
        "source_stats": source_stats,
    }

    if not delta_rows_list:
        logger.info("hard_negative_mining: no FHM metadata confounder pairs found with %s", metadata_path)
        return scipy.sparse.csr_matrix((0, train_csr.shape[1]), dtype=np.float32), [], stats

    delta_csr = scipy.sparse.vstack(delta_rows_list, format="csr").astype(np.float32)
    logger.info(
        "hard_negative_mining: built %d synthetic rows from %d FHM metadata pairs; source_stats=%s",
        len(delta_labels), len(seen_pairs), source_stats,
    )
    return delta_csr, delta_labels, stats


def run_validate(cfg: ProbeValidateConfig) -> None:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
    except ImportError as e:
        raise ImportError("scikit-learn is required. pip install scikit-learn scipy") from e

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cache = Path(cfg.train_cache_dir)
    val_cache = Path(cfg.val_cache_dir)
    baseline_path = Path(cfg.baseline_predictions_path) if cfg.baseline_predictions_path else None

    assert train_cache.exists(), f"Train cache not found: {train_cache}"
    if not val_cache.exists():
        raise FileNotFoundError(
            f"Validation residual cache not found: {val_cache}\n"
            "Run slurm/cache_dense_validation.sub then slurm/cache_residual_validation.sub first."
        )
    if baseline_path is not None:
        assert baseline_path.exists(), f"Baseline predictions file not found: {baseline_path}"

    # All val cache dirs (primary + extras)
    all_val_caches = [val_cache]
    for extra_dir in cfg.extra_val_cache_dirs:
        p = Path(extra_dir)
        if p.exists():
            all_val_caches.append(p)
        else:
            logger.warning("Extra val cache dir not found, skipping: %s", extra_dir)

    device = resolve_visible_device(cfg.device)

    logger.info("Loading SAE from %s", cfg.residual_sae_path)
    sae = load_residual_sae(cfg.residual_sae_path, device=device)
    stats = load_residual_stats(cfg.residual_sae_path)
    n_features = sae.num_latents
    logger.info("SAE num_latents=%d", n_features)

    datasets = set(cfg.datasets)

    # ── Load train manifest and extract features ────────────────────────────
    logger.info("Loading train manifest from %s", train_cache)
    train_rows_all = _read_jsonl(train_cache / "manifest.jsonl")
    train_rows = [
        r for r in train_rows_all
        if r.get("dataset_name") in datasets
        and r.get("split") == "train"
        and _row_gold_label(r) is not None
    ]
    logger.info("Train manifest: %d rows", len(train_rows))
    assert train_rows, f"No train rows matched from {train_cache}"

    # ── Model-agree filter for generated-token probes ───────────────────────
    # For datasets in model_agree_train_filter_datasets and token_selection=generated,
    # drop rows where model's own generated label != gold. These rows have SAE features
    # from tokens where the model reasoned incorrectly, creating contradictory training
    # signal: the features look like class A but are labelled as class B.
    active_token_selections = _required_token_selections(
        cfg.concat_token_selections,
        cfg.interaction_token_selection_pairs,
    ) or [cfg.token_selection]
    filter_datasets = set(cfg.model_agree_train_filter_datasets)
    uses_generated = any("generated" in ts for ts in active_token_selections)
    if filter_datasets and uses_generated:
        before = len(train_rows)
        filtered = []
        dropped_by_dataset: Counter = Counter()
        kept_by_dataset: Counter = Counter()
        for r in train_rows:
            ds = r.get("dataset_name", "")
            if ds not in filter_datasets:
                filtered.append(r)
                continue
            model_label = _parse_model_generated_label(r)
            gold_label = _row_gold_label(r)
            if model_label is None or model_label == gold_label:
                filtered.append(r)
                kept_by_dataset[ds] += 1
            else:
                dropped_by_dataset[ds] += 1
        train_rows = filtered
        logger.info(
            "model_agree_train_filter: kept %d / %d train rows (dropped %d contradictory). "
            "kept_by_dataset=%s dropped_by_dataset=%s",
            len(train_rows), before, before - len(train_rows),
            dict(kept_by_dataset), dict(dropped_by_dataset),
        )

    _extract_common_kw = dict(
        aggregation=cfg.aggregation,
        batch_size=cfg.batch_size,
        device=device,
        n_features=n_features,
    )

    if cfg.concat_token_selections or cfg.interaction_token_selection_pairs:
        required_token_selections = _required_token_selections(
            cfg.concat_token_selections,
            cfg.interaction_token_selection_pairs,
        )
        logger.info(
            "Extracting train features (concat_token_selections=%s, interaction_pairs=%s, aggregation=%s)...",
            cfg.concat_token_selections, cfg.interaction_token_selection_pairs, cfg.aggregation,
        )
        train_feature_sets = {
            ts: _extract_features(train_rows, train_cache, sae, stats,
                                  token_selection=ts, **_extract_common_kw)
            for ts in required_token_selections
        }
        train_keys, train_csr = _build_feature_blocks(
            train_feature_sets,
            cfg.concat_token_selections,
            cfg.interaction_token_selection_pairs,
        )
    else:
        logger.info("Extracting train features (token_selection=%s, aggregation=%s)...",
                    cfg.token_selection, cfg.aggregation)
        train_keys, train_csr = _extract_features(
            train_rows, train_cache, sae, stats,
            token_selection=cfg.token_selection, **_extract_common_kw,
        )
    # Build lookup: sample_key -> index in train_keys
    train_key_to_idx: dict[str, int] = {k: i for i, k in enumerate(train_keys)}
    logger.info("Train CSR: %d x %d, nnz=%d", train_csr.shape[0], train_csr.shape[1], train_csr.nnz)

    # ── Load validation manifest and extract features ───────────────────────
    # Merge manifests from all val cache dirs (supports Crisis_HateMM in a separate cache).
    # IMPORTANT: shard filenames are NOT globally unique across different cache dirs.
    # Tag each row with its cache dir path so _extract_features_multi can route correctly.
    val_rows_all: list[dict[str, Any]] = []
    for vc in all_val_caches:
        vc_rows = _read_jsonl(vc / "manifest.jsonl")
        for r in vc_rows:
            r["_cache_dir"] = str(vc)  # tag for routing in _extract_features_multi
        val_rows_all.extend(vc_rows)
        logger.info("Loaded %d rows from val cache %s", len(vc_rows), vc)

    val_rows = [
        r for r in val_rows_all
        if r.get("dataset_name") in datasets
        and r.get("split") in set(cfg.eval_splits)
        and _row_gold_label(r) is not None
    ]
    logger.info("Val manifest total: %d rows", len(val_rows))
    assert val_rows, f"No validation rows matched from {all_val_caches}"

    _extract_multi_common_kw = dict(
        sae=sae, stats=stats,
        aggregation=cfg.aggregation,
        batch_size=cfg.batch_size,
        device=device,
        n_features=n_features,
    )

    if cfg.concat_token_selections or cfg.interaction_token_selection_pairs:
        required_token_selections = _required_token_selections(
            cfg.concat_token_selections,
            cfg.interaction_token_selection_pairs,
        )
        logger.info(
            "Extracting val features (concat_token_selections=%s, interaction_pairs=%s)...",
            cfg.concat_token_selections, cfg.interaction_token_selection_pairs,
        )
        val_feature_sets = {
            ts: _extract_features_multi(val_rows, token_selection=ts, **_extract_multi_common_kw)
            for ts in required_token_selections
        }
        val_keys, val_csr = _build_feature_blocks(
            val_feature_sets,
            cfg.concat_token_selections,
            cfg.interaction_token_selection_pairs,
        )
    else:
        logger.info("Extracting val features...")
        val_keys, val_csr = _extract_features_multi(
            val_rows, token_selection=cfg.token_selection, **_extract_multi_common_kw,
        )
    val_key_to_idx: dict[str, int] = {k: i for i, k in enumerate(val_keys)}
    val_key_to_row: dict[str, dict[str, Any]] = {_row_sample_key(r): r for r in val_rows}
    logger.info("Val CSR: %d x %d, nnz=%d", val_csr.shape[0], val_csr.shape[1], val_csr.nnz)

    if cfg.save_feature_matrix:
        feature_matrix_path = output_dir / "feature_matrix.npz"
        _write_feature_matrix(feature_matrix_path, val_keys, val_csr)
        logger.info("Validation feature matrix written to %s", feature_matrix_path)

    # ── Load baseline predictions ────────────────────────────────────────────
    baseline_rows = [] if baseline_path is None else _read_jsonl(baseline_path)
    # Index by (dataset_name, task_name, sample_id) → prediction_label and label_scores
    baseline_pred: dict[str, str] = {}
    baseline_label_scores: dict[str, dict[str, float]] = {}
    for br in baseline_rows:
        key = f"{br['dataset_name']}::{br['task_name']}::{br['sample_id']}"
        baseline_pred[key] = br.get("prediction_label", "")
        if br.get("label_scores"):
            baseline_label_scores[key] = br["label_scores"]
    logger.info("Baseline predictions: %d entries", len(baseline_pred))

    # ── Group train rows by task ─────────────────────────────────────────────
    task_train: dict[str, list[tuple[int, str]]] = defaultdict(list)  # task_key -> [(idx, label)]
    for row in train_rows:
        sk = _row_sample_key(row)
        if sk not in train_key_to_idx:
            continue
        task_key = _row_task_key(row)
        label = _row_gold_label(row)
        task_train[task_key].append((train_key_to_idx[sk], label))

    # ── Group val rows by task ──────────────────────────────────────────────
    task_val: dict[str, list[tuple[int, str, str]]] = defaultdict(list)  # task_key -> [(idx, label, sample_key)]
    for row in val_rows:
        sk = _row_sample_key(row)
        if sk not in val_key_to_idx:
            continue
        task_key = _row_task_key(row)
        label = _row_gold_label(row)
        task_val[task_key].append((val_key_to_idx[sk], label, sk))

    # ── Fit probes and predict ───────────────────────────────────────────────
    all_task_metrics: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    pred_output: list[dict[str, Any]] = []

    for task_key in sorted(task_train.keys()):
        if task_key not in task_val:
            logger.warning("Task %s has train rows but no val rows — skipping", task_key)
            continue

        tr_entries = task_train[task_key]
        va_entries = task_val[task_key]

        tr_indices = [e[0] for e in tr_entries]
        tr_labels = [e[1] for e in tr_entries]
        va_indices = [e[0] for e in va_entries]
        va_labels = [e[1] for e in va_entries]
        va_sample_keys = [e[2] for e in va_entries]

        n_train = len(tr_indices)
        n_val = len(va_indices)
        task_ds = task_key.split("::")[0]

        tr_counts = Counter(tr_labels)
        if any(c < cfg.min_class_samples for c in tr_counts.values()):
            logger.warning("Skipping %s: some classes have < %d train samples", task_key, cfg.min_class_samples)
            continue

        logger.info("Task %s: train=%d, val=%d", task_key, n_train, n_val)

        X_train = train_csr[tr_indices]
        X_val = val_csr[va_indices]
        hard_negative_stats = None

        # ── Hard-negative mining: augment train with text-confounder delta pairs ──
        if task_ds in set(cfg.hard_negative_mining_datasets):
            import scipy.sparse
            if task_ds == "Hateful_Memes_Dataset" and cfg.fhm_confounder_metadata_path:
                delta_csr, delta_labels_raw, hard_negative_stats = _build_fhm_metadata_hard_negative_pairs(
                    train_rows,
                    train_csr,
                    train_key_to_idx,
                    Path(cfg.fhm_confounder_metadata_path),
                    cfg.fhm_confounder_pair_sources,
                )
            else:
                delta_csr, delta_labels_raw, hard_negative_stats = _build_hard_negative_pairs(
                    train_rows, train_csr, train_key_to_idx, dataset_name=task_ds,
                )
            if delta_labels_raw:
                X_train = scipy.sparse.vstack([X_train, delta_csr], format="csr")
                tr_labels = list(tr_labels) + delta_labels_raw
                logger.info(
                    "hard_negative_mining: augmented %s train %d → %d rows",
                    task_key, n_train, len(tr_labels),
                )

        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        le = LabelEncoder()
        y_train = le.fit_transform(tr_labels)
        y_val = le.transform(va_labels) if set(va_labels) <= set(le.classes_) else None
        classes = le.classes_.tolist()

        # Per-dataset C override (match by dataset_name prefix in task_key)
        effective_C = cfg.per_task_C_overrides.get(task_ds, cfg.C)
        effective_solver = (
            "liblinear" if n_train > cfg.large_dataset_solver_threshold else cfg.solver
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cfg.classifier_type == "mlp":
                from sklearn.neural_network import MLPClassifier
                logger.info(
                    "Fitting MLP probe: classes=%s, hidden=%d, dropout=%.2f, alpha=%g",
                    classes, cfg.mlp_hidden_size, cfg.mlp_dropout, cfg.mlp_alpha,
                )
                # MLPClassifier needs dense input; toarray() is fine for FHM (~650 MB)
                X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
                X_val_dense   = X_val.toarray()   if hasattr(X_val,   "toarray") else X_val
                clf = MLPClassifier(
                    hidden_layer_sizes=(cfg.mlp_hidden_size,),
                    activation="relu",
                    alpha=cfg.mlp_alpha,
                    max_iter=cfg.mlp_max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=42,
                    verbose=False,
                )
                clf.fit(X_train_dense, y_train)
                y_pred_enc = clf.predict(X_val_dense)
            else:
                logger.info("Fitting probe: classes=%s, solver=%s, C=%g", classes, effective_solver, effective_C)
                base_clf = LogisticRegression(
                    C=effective_C,
                    max_iter=cfg.max_iter,
                    solver=effective_solver,
                    class_weight=cfg.class_weight if cfg.class_weight != "None" else None,
                )
                # liblinear doesn't support multiclass natively — wrap with OVR
                if effective_solver == "liblinear" and len(classes) >= 3:
                    from sklearn.multiclass import OneVsRestClassifier
                    clf = OneVsRestClassifier(base_clf)
                else:
                    clf = base_clf
                clf.fit(X_train, y_train)
                y_pred_enc = clf.predict(X_val)

        if task_ds in set(cfg.save_probe_for_datasets):
            try:
                import re as _re

                import joblib as _joblib
                _joblib.dump(clf, output_dir / "probe.joblib")
                _joblib.dump(scaler, output_dir / "scaler.joblib")
                _joblib.dump(le, output_dir / "label_encoder.joblib")
                logger.info(
                    "Saved probe to %s, scaler to %s, label_encoder to %s",
                    output_dir / "probe.joblib",
                    output_dir / "scaler.joblib",
                    output_dir / "label_encoder.joblib",
                )
                _task_slug = _re.sub(r"[^A-Za-z0-9]+", "_", task_key).strip("_")
                _joblib.dump(clf, output_dir / f"probe__{_task_slug}.joblib")
                _joblib.dump(scaler, output_dir / f"scaler__{_task_slug}.joblib")
                _joblib.dump(le, output_dir / f"label_encoder__{_task_slug}.joblib")
                logger.info("Saved task-scoped probe copy with prefix probe__%s", _task_slug)
            except Exception as _save_err:
                logger.warning("Failed to save probe/scaler: %s", _save_err)

        # ── Shuffled-label control ────────────────────────────────────────────
        # Re-fit with randomly permuted training labels; genuine learning should
        # collapse toward chance accuracy (~1/n_classes for balanced data).
        shuffled_control: dict[str, Any] | None = None
        if cfg.shuffled_label_control:
            import random as _random
            _rng = _random.Random(cfg.control_seed)
            _sh_labels = list(tr_labels)
            _rng.shuffle(_sh_labels)
            _y_sh = le.transform(_sh_labels)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if cfg.classifier_type == "mlp":
                    _clf_sh = type(clf)(**clf.get_params())
                    _clf_sh.fit(X_train_dense, _y_sh)
                    _pred_sh = le.inverse_transform(_clf_sh.predict(X_val_dense)).tolist()
                else:
                    _clf_sh = type(clf)(**clf.get_params()) if not hasattr(clf, "estimator") else type(clf)(type(clf.estimator)(**clf.estimator.get_params()))
                    _clf_sh.fit(X_train, _y_sh)
                    _pred_sh = le.inverse_transform(_clf_sh.predict(X_val)).tolist()
            _sm = _per_class_metrics(va_labels, _pred_sh)
            shuffled_control = {
                "macro_f1": _sm["macro_f1"],
                "accuracy": _sm["accuracy"],
                "chance_f1": round(1.0 / len(classes), 4),
            }
            logger.info(
                "  Shuffled-label control %s: macro_f1=%.4f (chance=%.4f)",
                task_key, _sm["macro_f1"], 1.0 / len(classes),
            )

        # ── Column-permutation control ────────────────────────────────────────
        # Permute feature columns; destroys SAE latent-semantic assignments while
        # preserving sparsity, scale, and activation count per sample.
        # If real ≈ permuted, the specific SAE structure doesn't matter.
        column_perm_control: dict[str, Any] | None = None
        if cfg.column_permutation_control:
            import scipy.sparse as _sp
            _rng_np = np.random.default_rng(cfg.control_seed)
            _perm = _rng_np.permutation(train_csr.shape[1])
            _X_tr_perm = X_train[:, _perm] if hasattr(X_train, "shape") else X_train
            _X_va_perm = X_val[:, _perm] if hasattr(X_val, "shape") else X_val
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if cfg.classifier_type == "mlp":
                    _clf_p = type(clf)(**clf.get_params())
                    _X_tr_perm_d = _X_tr_perm.toarray() if hasattr(_X_tr_perm, "toarray") else _X_tr_perm
                    _X_va_perm_d = _X_va_perm.toarray() if hasattr(_X_va_perm, "toarray") else _X_va_perm
                    _clf_p.fit(_X_tr_perm_d, y_train)
                    _pred_p = le.inverse_transform(_clf_p.predict(_X_va_perm_d)).tolist()
                else:
                    _clf_p = type(clf)(**clf.get_params()) if not hasattr(clf, "estimator") else type(clf)(type(clf.estimator)(**clf.estimator.get_params()))
                    _clf_p.fit(_X_tr_perm, y_train)
                    _pred_p = le.inverse_transform(_clf_p.predict(_X_va_perm)).tolist()
            _pm = _per_class_metrics(va_labels, _pred_p)
            column_perm_control = {
                "macro_f1": _pm["macro_f1"],
                "accuracy": _pm["accuracy"],
                "real_macro_f1": None,  # filled below once probe_metrics is computed
            }
            logger.info(
                "  Column-permutation control %s: macro_f1=%.4f",
                task_key, _pm["macro_f1"],
            )

        # ── Random-features control ───────────────────────────────────────────
        # Replace SAE features with random sparse vectors of same L0 and scale.
        random_feat_control: dict[str, Any] | None = None
        if cfg.random_features_control:
            _rng_rf = np.random.default_rng(cfg.control_seed + 2)
            _X_tr_rand = _make_random_sparse_like(train_csr[tr_indices], _rng_rf)
            _X_va_rand = _make_random_sparse_like(val_csr[va_indices],   _rng_rf)
            _scaler_rf = MaxAbsScaler()
            _X_tr_rand = _scaler_rf.fit_transform(_X_tr_rand)
            _X_va_rand = _scaler_rf.transform(_X_va_rand)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if cfg.classifier_type == "mlp":
                    _clf_rf = type(clf)(**clf.get_params())
                    _clf_rf.fit(_X_tr_rand.toarray(), y_train)
                    _pred_rf = le.inverse_transform(_clf_rf.predict(_X_va_rand.toarray())).tolist()
                else:
                    _clf_rf = type(clf)(**clf.get_params()) if not hasattr(clf, "estimator") else type(clf)(type(clf.estimator)(**clf.estimator.get_params()))
                    _clf_rf.fit(_X_tr_rand, y_train)
                    _pred_rf = le.inverse_transform(_clf_rf.predict(_X_va_rand)).tolist()
            _rfm = _per_class_metrics(va_labels, _pred_rf)
            random_feat_control = {
                "macro_f1": _rfm["macro_f1"],
                "accuracy": _rfm["accuracy"],
                "chance_f1": round(1.0 / len(classes), 4),
            }
            logger.info(
                "  Random-features control %s: macro_f1=%.4f (real=later, chance=%.4f)",
                task_key, _rfm["macro_f1"], 1.0 / len(classes),
            )

        # ── Raw-residual-stream control ───────────────────────────────────────
        # Fit same probe on 2560-dim hidden states from dense cache (no SAE).
        raw_residual_control: dict[str, Any] | None = None
        if cfg.raw_residual_control_train_cache_dir and cfg.raw_residual_control_val_cache_dir:
            _dense_tr = Path(cfg.raw_residual_control_train_cache_dir)
            _dense_va = Path(cfg.raw_residual_control_val_cache_dir)
            if _dense_tr.exists() and _dense_va.exists():
                _tr_rows_raw = [r for r in train_rows if _row_sample_key(r) in {train_keys[i] for i in tr_indices}]
                _va_rows_raw = [r for r in val_rows  if _row_sample_key(r) in set(va_sample_keys)]
                _tr_keys_raw, _X_tr_raw = _extract_raw_features(
                    _tr_rows_raw, _dense_tr,
                    token_selection=cfg.token_selection, aggregation=cfg.aggregation,
                )
                _va_keys_raw, _X_va_raw = _extract_raw_features(
                    _va_rows_raw, _dense_va,
                    token_selection=cfg.token_selection, aggregation=cfg.aggregation,
                )
                if len(_tr_keys_raw) > 0 and len(_va_keys_raw) > 0:
                    # Align to task indices (tr_labels already aligned)
                    _key_to_i_tr = {k: i for i, k in enumerate(_tr_keys_raw)}
                    _key_to_i_va = {k: i for i, k in enumerate(_va_keys_raw)}
                    _tr_sel = [_key_to_i_tr[train_keys[i]] for i in tr_indices if train_keys[i] in _key_to_i_tr]
                    _va_sel = [_key_to_i_va[sk]              for sk in va_sample_keys if sk in _key_to_i_va]
                    if len(_tr_sel) == len(tr_indices) and len(_va_sel) == len(va_sample_keys):
                        _X_tr_rr = _X_tr_raw[_tr_sel]
                        _X_va_rr = _X_va_raw[_va_sel]
                        _scaler_rr = MaxAbsScaler()
                        _X_tr_rr = _scaler_rr.fit_transform(_X_tr_rr)
                        _X_va_rr = _scaler_rr.transform(_X_va_rr)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if cfg.classifier_type == "mlp":
                                _clf_rr = type(clf)(**clf.get_params())
                                _clf_rr.fit(_X_tr_rr, y_train)
                                _pred_rr = le.inverse_transform(_clf_rr.predict(_X_va_rr)).tolist()
                            else:
                                _clf_rr = type(clf)(**clf.get_params()) if not hasattr(clf, "estimator") else type(clf)(type(clf.estimator)(**clf.estimator.get_params()))
                                _clf_rr.fit(_X_tr_rr, y_train)
                                _pred_rr = le.inverse_transform(_clf_rr.predict(_X_va_rr)).tolist()
                        _rrm = _per_class_metrics(va_labels, _pred_rr)
                        raw_residual_control = {
                            "macro_f1": _rrm["macro_f1"],
                            "accuracy": _rrm["accuracy"],
                            "hidden_dim": _X_tr_raw.shape[1],
                            "n_tr_extracted": len(_tr_keys_raw),
                        }
                        logger.info(
                            "  Raw-residual control %s: macro_f1=%.4f (dim=%d)",
                            task_key, _rrm["macro_f1"], _X_tr_raw.shape[1],
                        )
                        # Save the raw-activation-only clf + scaler for a "Stage 2"
                        # deployment (probe on frozen raw hidden states, skipping the
                        # SAE entirely) -- see gemma3_residual_sae/meme_safety_probe.
                        if task_ds in set(cfg.save_probe_for_datasets):
                            try:
                                import joblib as _joblib
                                _joblib.dump(_clf_rr, output_dir / "probe_raw.joblib")
                                _joblib.dump(_scaler_rr, output_dir / "scaler_raw.joblib")
                                logger.info(
                                    "Saved raw-activation probe to %s, scaler to %s",
                                    output_dir / "probe_raw.joblib",
                                    output_dir / "scaler_raw.joblib",
                                )
                            except Exception as _save_raw_err:
                                logger.warning("Failed to save raw-activation probe/scaler: %s", _save_raw_err)
                    else:
                        logger.warning("Raw-residual control %s: key alignment failed (%d/%d tr, %d/%d va)",
                                       task_key, len(_tr_sel), len(tr_indices), len(_va_sel), len(va_sample_keys))
            else:
                logger.warning("raw_residual_control dirs not found: train=%s val=%s", _dense_tr, _dense_va)

        # ── Save fitted clf + scaler for causal ablation ──────────────────────
        if task_ds in set(cfg.save_probe_for_datasets):
            try:
                import joblib as _joblib
                _joblib.dump(clf, output_dir / "probe.joblib")
                _joblib.dump(scaler, output_dir / "scaler.joblib")
                _joblib.dump(le, output_dir / "label_encoder.joblib")
                logger.info(
                    "Saved probe to %s, scaler to %s, label_encoder to %s",
                    output_dir / "probe.joblib",
                    output_dir / "scaler.joblib",
                    output_dir / "label_encoder.joblib",
                )
            except Exception as _save_err:
                logger.warning("Failed to save probe/scaler: %s", _save_err)

        probe_preds = le.inverse_transform(y_pred_enc).tolist()

        # Predict probabilities for ensemble use (probe_proba: {class: prob})
        y_proba = None
        try:
            # MLP uses dense X_val; logistic uses sparse X_val (X_val_dense may not exist)
            _X_val_for_proba = X_val_dense if cfg.classifier_type == "mlp" else X_val
            y_proba = clf.predict_proba(_X_val_for_proba)
            probe_proba_list = [
                {le.classes_[j]: round(float(y_proba[i, j]), 6) for j in range(len(le.classes_))}
                for i in range(len(va_sample_keys))
            ]
        except Exception:
            probe_proba_list = [None] * len(va_sample_keys)

        probe_metrics = _per_class_metrics(va_labels, probe_preds)
        logger.info("Probe %s: accuracy=%.4f macro_f1=%.4f",
                    task_key, probe_metrics["accuracy"], probe_metrics["macro_f1"])

        # ── Threshold calibration (binary tasks only, opt-in per dataset) ────
        calibrated_metrics = None
        if (
            y_proba is not None
            and len(classes) == 2
            and task_ds in cfg.calibrate_threshold_datasets
        ):
            # FHM positive class is "hateful"; for generic binary tasks use the
            # class with lower gold count (minority = positive).
            gold_counts = Counter(va_labels)
            pos_class = min(gold_counts, key=lambda c: gold_counts[c])

            # Optional calibration split: use only the first `calibration_fraction`
            # of val samples for threshold selection to avoid overfit to the eval set.
            if cfg.calibration_fraction > 0:
                n_cal = max(1, int(len(va_labels) * cfg.calibration_fraction))
                cal_labels = va_labels[:n_cal]
                cal_proba  = y_proba[:n_cal]
                eval_labels = va_labels[n_cal:]
                eval_proba  = y_proba[n_cal:]
                best_t, _, _ = _calibrate_threshold(cal_labels, cal_proba, classes, pos_class)
                # Apply the calibration-selected threshold to the held-out eval subset
                pos_idx = classes.index(pos_class)
                best_cal_preds_eval = [
                    classes[pos_idx] if eval_proba[i, pos_idx] >= best_t
                    else classes[1 - pos_idx]
                    for i in range(len(eval_labels))
                ]
                calibrated_metrics = _per_class_metrics(eval_labels, best_cal_preds_eval)
                calibrated_metrics["threshold"] = round(best_t, 2)
                calibrated_metrics["calibration_n"] = n_cal
                calibrated_metrics["eval_n"] = len(eval_labels)
                logger.info(
                    "  Calibrated %s (pos=%s, t=%.2f, cal_n=%d, eval_n=%d): "
                    "macro_f1=%.4f [held-out subset, not full val]",
                    task_key, pos_class, best_t, n_cal, len(eval_labels),
                    calibrated_metrics["macro_f1"],
                )
            else:
                best_t, best_f1, best_cal_preds = _calibrate_threshold(
                    va_labels, y_proba, classes, pos_class,
                )
                if best_cal_preds:
                    calibrated_metrics = _per_class_metrics(va_labels, best_cal_preds)
                    calibrated_metrics["threshold"] = round(best_t, 2)
                    logger.info(
                        "  Calibrated %s (pos=%s, t=%.2f): macro_f1=%.4f (raw=%.4f, delta=+%.4f)",
                        task_key, pos_class, best_t, best_f1,
                        probe_metrics["macro_f1"], best_f1 - probe_metrics["macro_f1"],
                    )

        # Gather baseline predictions for this task's val samples
        baseline_preds_for_task = [baseline_pred.get(sk, None) for sk in va_sample_keys]
        # Only evaluate baseline where a prediction exists and is in the known label set
        valid_base = [(g, b) for g, b in zip(va_labels, baseline_preds_for_task) if b is not None]
        if valid_base:
            base_gold, base_pred_list = zip(*valid_base)
            baseline_metrics = _per_class_metrics(list(base_gold), list(base_pred_list))
        else:
            baseline_metrics = None

        ds, tn, fn = task_key.split("::", maxsplit=2)
        task_result: dict[str, Any] = {
            "task_key": task_key,
            "dataset_name": ds,
            "task_name": tn,
            "field_name": fn,
            "n_train": n_train,
            "n_val": n_val,
            "classes": classes,
            "probe_metrics": probe_metrics,
            "calibrated_metrics": calibrated_metrics,
            "baseline_metrics": baseline_metrics,
            "hard_negative_stats": hard_negative_stats,
            "shuffled_label_control": shuffled_control,
            "column_permutation_control": (
                {**column_perm_control, "real_macro_f1": probe_metrics["macro_f1"]}
                if column_perm_control else None
            ),
            "random_features_control": (
                {**random_feat_control, "real_macro_f1": probe_metrics["macro_f1"]}
                if random_feat_control else None
            ),
            "raw_residual_control": (
                {**raw_residual_control, "real_macro_f1": probe_metrics["macro_f1"]}
                if raw_residual_control else None
            ),
            "macro_f1_delta": (
                round(probe_metrics["macro_f1"] - baseline_metrics["macro_f1"], 4)
                if baseline_metrics else None
            ),
        }
        all_task_metrics[task_key] = task_result
        comparison_rows.append(task_result)

        # Write per-sample predictions (including probabilities for ensemble)
        for sk, gold, pred, proba in zip(va_sample_keys, va_labels, probe_preds, probe_proba_list):
            base = baseline_pred.get(sk)
            base_scores = baseline_label_scores.get(sk)
            pred_output.append({
                "sample_key": sk,
                "task_key": task_key,
                "gold_label": gold,
                "probe_pred": pred,
                "probe_proba": proba,
                "baseline_pred": base,
                "baseline_label_scores": base_scores,
                "probe_correct": pred == gold,
                "baseline_correct": base == gold if base else None,
            })

        # Also run binary collapse if applicable
        if cfg.binary_collapse_nothate and "NotHate" in classes:
            def _to_bin(lst: list[str]) -> list[str]:
                return ["NotHate" if x == "NotHate" else "Hate" for x in lst]
            le_bin = LabelEncoder()
            y_bin_tr = le_bin.fit_transform(_to_bin(tr_labels))
            bin_val_labels = _to_bin(va_labels)
            bin_probe_preds = [
                "NotHate" if p == "NotHate" else "Hate" for p in probe_preds
            ]
            probe_bin_metrics = _per_class_metrics(bin_val_labels, bin_probe_preds)
            base_bin = [
                ("NotHate" if b == "NotHate" else "Hate") if b else None
                for b in baseline_preds_for_task
            ]
            valid_base_bin = [(g, b) for g, b in zip(bin_val_labels, base_bin) if b is not None]
            if valid_base_bin:
                bg, bp = zip(*valid_base_bin)
                baseline_bin_metrics = _per_class_metrics(list(bg), list(bp))
            else:
                baseline_bin_metrics = None

            bin_key = task_key + " [binary]"
            bin_result: dict[str, Any] = {
                "task_key": bin_key,
                "dataset_name": ds,
                "task_name": tn,
                "field_name": fn + " [binary]",
                "n_train": n_train,
                "n_val": n_val,
                "classes": ["Hate", "NotHate"],
                "probe_metrics": probe_bin_metrics,
                "baseline_metrics": baseline_bin_metrics,
                "macro_f1_delta": (
                    round(probe_bin_metrics["macro_f1"] - baseline_bin_metrics["macro_f1"], 4)
                    if baseline_bin_metrics else None
                ),
            }
            all_task_metrics[bin_key] = bin_result
            comparison_rows.append(bin_result)

        # Harmful binary collapse: "somewhat harmful" + "very harmful" → "harmful"
        if cfg.binary_collapse_harmful and "somewhat harmful" in classes and "very harmful" in classes:
            def _to_harm_bin(lst: list[str]) -> list[str]:
                return ["not harmful" if x == "not harmful" else "harmful" for x in lst]
            harm_val_labels = _to_harm_bin(va_labels)
            harm_probe_preds = _to_harm_bin(probe_preds)
            probe_harm_metrics = _per_class_metrics(harm_val_labels, harm_probe_preds)
            base_harm = [
                _to_harm_bin([b])[0] if b else None for b in baseline_preds_for_task
            ]
            valid_base_harm = [(g, b) for g, b in zip(harm_val_labels, base_harm) if b is not None]
            if valid_base_harm:
                hg, hp = zip(*valid_base_harm)
                baseline_harm_metrics = _per_class_metrics(list(hg), list(hp))
            else:
                baseline_harm_metrics = None

            harm_key = task_key + " [harm_binary]"
            harm_result: dict[str, Any] = {
                "task_key": harm_key,
                "dataset_name": ds,
                "task_name": tn,
                "field_name": fn + " [harm_binary]",
                "n_train": n_train,
                "n_val": n_val,
                "classes": ["harmful", "not harmful"],
                "probe_metrics": probe_harm_metrics,
                "baseline_metrics": baseline_harm_metrics,
                "macro_f1_delta": (
                    round(probe_harm_metrics["macro_f1"] - baseline_harm_metrics["macro_f1"], 4)
                    if baseline_harm_metrics else None
                ),
            }
            all_task_metrics[harm_key] = harm_result
            comparison_rows.append(harm_result)

    # ── Write outputs ─────────────────────────────────────────────────────────

    # ── Hierarchical MMHS probe (Step 4) ─────────────────────────────────────
    # Two-stage: (1) binary Hate/NotHate, (2) hate-type on binary-Hate predictions.
    mmhs_task_key = "MMHS150K::hate_category::label"
    if cfg.hierarchical_mmhs and mmhs_task_key in task_train and mmhs_task_key in task_val:
        _run_hierarchical_mmhs(
            mmhs_task_key, task_train, task_val,
            train_csr, val_csr,
            cfg, all_task_metrics, comparison_rows, pred_output,
            baseline_pred,
        )

    # ── MMHS one-vs-rest per-class probes (Step 4b) ───────────────────────────
    if cfg.mmhs_ovr and mmhs_task_key in task_train and mmhs_task_key in task_val:
        _run_mmhs_ovr(
            mmhs_task_key, task_train, task_val,
            train_csr, val_csr,
            cfg, all_task_metrics, comparison_rows,
        )

    # ── FHM benign confounder analysis (Step 5) ───────────────────────────────
    fhm_task_key = "Hateful_Memes_Dataset::hateful_memes_binary::label"
    if cfg.fhm_confounder_analysis and fhm_task_key in all_task_metrics:
        _write_fhm_confounder_report(
            fhm_task_key, pred_output, val_key_to_row, output_dir,
        )

    metrics_path = output_dir / "probe_val_metrics.json"
    metrics_path.write_text(json.dumps(all_task_metrics, indent=2))
    logger.info("Metrics written to %s", metrics_path)

    preds_path = output_dir / "probe_val_predictions.jsonl"
    with preds_path.open("w") as fd:
        for row in pred_output:
            fd.write(json.dumps(row) + "\n")
    logger.info("Per-sample predictions written to %s", preds_path)

    # Print summary table
    logger.info("=" * 72)
    logger.info("%-55s  %6s  %6s  %6s", "Task", "Probe", "Base", "Delta")
    logger.info("-" * 72)
    for r in sorted(comparison_rows, key=lambda x: -(x["probe_metrics"]["macro_f1"])):
        probe_f1 = r["probe_metrics"]["macro_f1"]
        base_f1 = r["baseline_metrics"]["macro_f1"] if r["baseline_metrics"] else float("nan")
        delta = r["macro_f1_delta"] if r["macro_f1_delta"] is not None else float("nan")
        logger.info("%-55s  %.4f  %.4f  %+.4f", r["task_key"][:55], probe_f1, base_f1, delta)
    logger.info("=" * 72)

    if cfg.plot:
        _plot_comparison(comparison_rows, output_dir)


def _run_mmhs_ovr(
    task_key: str,
    task_train: dict[str, list],
    task_val: dict[str, list],
    train_csr,
    val_csr,
    cfg: "ProbeValidateConfig",
    all_task_metrics: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
) -> None:
    pass
    try:
        import warnings
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import MaxAbsScaler
    except ImportError:
        logger.warning("scikit-learn unavailable – skipping MMHS OvR probes")
        return

    mmhs_task_key = "MMHS150K::hate_category::label"
    if mmhs_task_key not in task_train or mmhs_task_key not in task_val:
        return

    tr_entries = task_train[mmhs_task_key]
    va_entries = task_val[mmhs_task_key]
    tr_indices = [e[0] for e in tr_entries]
    tr_labels  = [e[1] for e in tr_entries]
    va_indices = [e[0] for e in va_entries]
    va_labels  = [e[1] for e in va_entries]

    all_classes = sorted(set(tr_labels) | set(va_labels))
    X_tr_all = train_csr[tr_indices]
    X_va_all = val_csr[va_indices]
    n_train_total = len(tr_indices)

    effective_solver = (
        "liblinear" if n_train_total > cfg.large_dataset_solver_threshold else cfg.solver
    )

    for cls in all_classes:
        y_tr_bin = [1 if l == cls else 0 for l in tr_labels]
        y_va_bin = [1 if l == cls else 0 for l in va_labels]
        n_pos_tr = sum(y_tr_bin)
        n_pos_va = sum(y_va_bin)
        if n_pos_tr < cfg.min_class_samples:
            logger.info("MMHS OvR %s: only %d train positives, skipping", cls, n_pos_tr)
            continue

        scaler = MaxAbsScaler()
        X_tr = scaler.fit_transform(X_tr_all)
        X_va = scaler.transform(X_va_all)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                C=cfg.C, max_iter=cfg.max_iter,
                solver=effective_solver,
                class_weight="balanced",
            )
            clf.fit(X_tr, y_tr_bin)

        y_proba = clf.predict_proba(X_va)[:, 1]   # P(class=cls)

        # OvR AUC via rank statistic
        pos_s = [s for s, g in zip(y_proba, y_va_bin) if g == 1]
        neg_s = [s for s, g in zip(y_proba, y_va_bin) if g == 0]
        if pos_s and neg_s:
            n_conc = sum(1 for p in pos_s for n in neg_s if p > n) + \
                     sum(0.5 for p in pos_s for n in neg_s if p == n)
            auc = round(n_conc / (len(pos_s) * len(neg_s)), 4)
        else:
            auc = float("nan")

        # Threshold sweep
        best_t, best_f1, best_pr, best_re = 0.5, 0.0, 0.0, 0.0
        for t_int in range(5, 96):
            t = t_int / 100.0
            tp = sum(1 for s, g in zip(y_proba, y_va_bin) if s >= t and g == 1)
            fp = sum(1 for s, g in zip(y_proba, y_va_bin) if s >= t and g == 0)
            fn = sum(1 for s, g in zip(y_proba, y_va_bin) if s < t and g == 1)
            pr = tp / (tp + fp) if tp + fp > 0 else 0.0
            re = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * pr * re / (pr + re) if pr + re > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t, best_pr, best_re = f1, t, pr, re

        ovr_key = f"{task_key} [ovr_{cls}]"
        result: dict[str, Any] = {
            "task_key": ovr_key,
            "dataset_name": "MMHS150K",
            "n_train_pos": n_pos_tr,
            "n_train_total": n_train_total,
            "n_val_pos": n_pos_va,
            "n_val": len(va_indices),
            "ovr_class": cls,
            "auc": auc,
            "calibrated_threshold": round(best_t, 2),
            "calibrated_f1": round(best_f1, 4),
            "calibrated_precision": round(best_pr, 4),
            "calibrated_recall": round(best_re, 4),
            "note": (
                "high-variance: fewer than 50 positive val samples"
                if n_pos_va < 50 else ""
            ),
        }
        all_task_metrics[ovr_key] = result
        logger.info(
            "MMHS OvR %-12s n_tr_pos=%4d n_va_pos=%3d AUC=%.3f t=%.2f cal_F1=%.4f P=%.3f R=%.3f%s",
            cls, n_pos_tr, n_pos_va, auc, best_t, best_f1, best_pr, best_re,
            "  [low-n]" if n_pos_va < 50 else "",
        )


def _run_hierarchical_mmhs(
    task_key: str,
    task_train: dict[str, list],
    task_val: dict[str, list],
    train_csr,
    val_csr,
    cfg: "ProbeValidateConfig",
    all_task_metrics: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    pred_output: list[dict[str, Any]],
    baseline_pred: dict[str, str],
) -> None:
    pass
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
    except ImportError:
        logger.warning("scikit-learn not available — skipping hierarchical MMHS probe")
        return
    import warnings

    tr_entries = task_train[task_key]
    va_entries = task_val[task_key]
    tr_indices = [e[0] for e in tr_entries]
    tr_labels = [e[1] for e in tr_entries]
    va_indices = [e[0] for e in va_entries]
    va_labels = [e[1] for e in va_entries]
    va_sample_keys = [e[2] for e in va_entries]

    X_train_all = train_csr[tr_indices]
    X_val_all = val_csr[va_indices]

    # ── Stage 1: binary NotHate vs Hate ──────────────────────────────────────
    def _to_binary(labels: list[str]) -> list[str]:
        return ["NotHate" if lbl == "NotHate" else "Hate" for lbl in labels]

    bin_tr_labels = _to_binary(tr_labels)
    bin_va_labels = _to_binary(va_labels)

    scaler1 = MaxAbsScaler()
    X_tr1 = scaler1.fit_transform(X_train_all)
    X_va1 = scaler1.transform(X_val_all)

    le1 = LabelEncoder()
    y_tr1 = le1.fit_transform(bin_tr_labels)

    effective_solver = "liblinear" if len(tr_indices) > cfg.large_dataset_solver_threshold else cfg.solver
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf1 = LogisticRegression(
            C=cfg.C, max_iter=cfg.max_iter, solver=effective_solver,
            class_weight=cfg.class_weight if cfg.class_weight != "None" else None,
        )
        clf1.fit(X_tr1, y_tr1)

    stage1_preds_va = le1.inverse_transform(clf1.predict(X_va1)).tolist()
    stage1_proba_va = clf1.predict_proba(X_va1)

    # ── Stage 2: 5-class hate-type on hate train samples only ────────────────
    hate_tr_mask = [lbl != "NotHate" for lbl in tr_labels]
    hate_tr_idx_local = [i for i, m in enumerate(hate_tr_mask) if m]
    hate_tr_global = [tr_indices[i] for i in hate_tr_idx_local]
    hate_tr_labels = [tr_labels[i] for i in hate_tr_idx_local]

    hate_class_counts = Counter(hate_tr_labels)
    if any(c < cfg.min_class_samples for c in hate_class_counts.values()):
        logger.warning("Hierarchical Stage 2: some hate-type classes have < %d train samples — skipping", cfg.min_class_samples)
        return

    X_tr2 = train_csr[hate_tr_global]
    scaler2 = MaxAbsScaler()
    X_tr2 = scaler2.fit_transform(X_tr2)

    le2 = LabelEncoder()
    y_tr2 = le2.fit_transform(hate_tr_labels)
    hate_classes = le2.classes_.tolist()

    effective_solver2 = "liblinear" if len(hate_tr_global) > cfg.large_dataset_solver_threshold else cfg.solver
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if effective_solver2 == "liblinear" and len(hate_classes) >= 3:
            from sklearn.multiclass import OneVsRestClassifier
            clf2 = OneVsRestClassifier(LogisticRegression(
                C=cfg.C, max_iter=cfg.max_iter, solver=effective_solver2,
                class_weight=cfg.class_weight if cfg.class_weight != "None" else None,
            ))
        else:
            clf2 = LogisticRegression(
                C=cfg.C, max_iter=cfg.max_iter, solver=effective_solver2,
                class_weight=cfg.class_weight if cfg.class_weight != "None" else None,
            )
        clf2.fit(X_tr2, y_tr2)

    # ── Hierarchical prediction on val ────────────────────────────────────────
    # Run Stage 2 on ALL val samples, then use Stage 1 to choose.
    X_va2 = scaler2.transform(X_val_all)
    stage2_preds_va = le2.inverse_transform(clf2.predict(X_va2)).tolist()

    hier_preds = [
        s1 if s1 == "NotHate" else s2
        for s1, s2 in zip(stage1_preds_va, stage2_preds_va)
    ]

    hier_metrics = _per_class_metrics(va_labels, hier_preds)
    bin_metrics_hier = _per_class_metrics(bin_va_labels, [_to_binary([p])[0] for p in hier_preds])

    # Compare to baseline
    baseline_preds_for_task = [baseline_pred.get(sk) for sk in va_sample_keys]
    valid_base = [(g, b) for g, b in zip(va_labels, baseline_preds_for_task) if b is not None]
    if valid_base:
        bg, bp = zip(*valid_base)
        baseline_metrics_hier = _per_class_metrics(list(bg), list(bp))
    else:
        baseline_metrics_hier = None

    logger.info(
        "Hierarchical MMHS: macro_f1=%.4f (binary stage1=%.4f), baseline=%.4f",
        hier_metrics["macro_f1"],
        bin_metrics_hier["macro_f1"],
        baseline_metrics_hier["macro_f1"] if baseline_metrics_hier else float("nan"),
    )

    hier_key = task_key + " [hierarchical]"
    hier_result: dict[str, Any] = {
        "task_key": hier_key,
        "dataset_name": "MMHS150K",
        "task_name": "hate_category",
        "field_name": "label [hierarchical]",
        "n_train": len(tr_indices),
        "n_val": len(va_indices),
        "classes": sorted(set(va_labels)),
        "stage1_classes": le1.classes_.tolist(),
        "stage2_classes": hate_classes,
        "n_hate_train": len(hate_tr_global),
        "probe_metrics": hier_metrics,
        "stage1_binary_metrics": bin_metrics_hier,
        "baseline_metrics": baseline_metrics_hier,
        "macro_f1_delta": (
            round(hier_metrics["macro_f1"] - baseline_metrics_hier["macro_f1"], 4)
            if baseline_metrics_hier else None
        ),
    }
    all_task_metrics[hier_key] = hier_result
    comparison_rows.append(hier_result)


def _write_fhm_confounder_report(
    task_key: str,
    pred_output: list[dict[str, Any]],
    val_key_to_row: dict[str, Any],
    output_dir: Path,
) -> None:
    pass
    fhm_rows = [r for r in pred_output if r.get("task_key") == task_key]
    if not fhm_rows:
        return

    report: list[dict[str, Any]] = []
    counts: dict[str, int] = {"probe_wrong_base_right": 0, "probe_right_base_wrong": 0,
                               "both_wrong": 0, "both_right": 0}
    for row in fhm_rows:
        probe_ok = row.get("probe_correct", False)
        base_ok = row.get("baseline_correct")
        if base_ok is None:
            continue
        if not probe_ok and base_ok:
            category = "probe_wrong_base_right"
        elif probe_ok and not base_ok:
            category = "probe_right_base_wrong"
        elif not probe_ok and not base_ok:
            category = "both_wrong"
        else:
            category = "both_right"
        counts[category] += 1

        raw = val_key_to_row.get(row["sample_key"], {})
        report.append({
            "sample_key": row["sample_key"],
            "gold": row["gold_label"],
            "probe_pred": row["probe_pred"],
            "baseline_pred": row.get("baseline_pred"),
            "category": category,
            "meme_text": raw.get("meme_text", ""),
            "image_path": str(raw.get("image_path", "")),
            "probe_proba": row.get("probe_proba"),
        })

    # Sort by category then alphabetically for readability
    report.sort(key=lambda r: (r["category"], r["sample_key"]))

    report_path = output_dir / "fhm_confounder_analysis.jsonl"
    with report_path.open("w") as fd:
        for r in report:
            fd.write(json.dumps(r) + "\n")

    summary_path = output_dir / "fhm_confounder_summary.json"
    n = len(fhm_rows)
    summary = {
        "total_samples": n,
        "counts": counts,
        "rates": {k: round(v / n, 4) for k, v in counts.items()},
        "notes": {
            "probe_wrong_base_right": (
                "Probe fails but baseline succeeds — probe may over-rely on visual SAE "
                "features and miss the text+image interaction needed for benign confounders."
            ),
            "probe_right_base_wrong": (
                "Probe succeeds but baseline fails — SAE features capture signal the "
                "LLM constrained decoder misses. Likely text-heavy or subtle visual cues."
            ),
            "both_wrong": (
                "Both systems fail — these are the hard benign confounders where neither "
                "visual nor LLM-decoded text alone is sufficient."
            ),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "FHM confounder analysis: total=%d, probe_wrong_base_right=%d, "
        "probe_right_base_wrong=%d, both_wrong=%d, both_right=%d",
        n, counts["probe_wrong_base_right"], counts["probe_right_base_wrong"],
        counts["both_wrong"], counts["both_right"],
    )
    logger.info("FHM confounder report written to %s", report_path)


def _plot_comparison(comparison_rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    # ── 1. Macro F1 comparison bar chart (probe vs baseline) ─────────────────
    rows_with_base = [r for r in comparison_rows if r["baseline_metrics"] is not None]
    rows_sorted = sorted(rows_with_base, key=lambda r: -r["probe_metrics"]["macro_f1"])

    if rows_sorted:
        task_names = [r["task_key"].replace("::", "\n") for r in rows_sorted]
        probe_f1s = [r["probe_metrics"]["macro_f1"] for r in rows_sorted]
        base_f1s = [r["baseline_metrics"]["macro_f1"] for r in rows_sorted]
        deltas = [r["macro_f1_delta"] for r in rows_sorted]

        x = np.arange(len(task_names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(10, len(task_names) * 1.6), 6))
        bars_p = ax.bar(x - width / 2, probe_f1s, width, label="SAE Probe", color="steelblue")
        bars_b = ax.bar(x + width / 2, base_f1s, width, label="Baseline (constrained decode)", color="darkorange")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

        for bar_p, bar_b, delta in zip(bars_p, bars_b, deltas):
            ax.text(bar_p.get_x() + bar_p.get_width() / 2, bar_p.get_height() + 0.01,
                    f"{bar_p.get_height():.3f}", ha="center", va="bottom", fontsize=7.5, color="steelblue")
            ax.text(bar_b.get_x() + bar_b.get_width() / 2, bar_b.get_height() + 0.01,
                    f"{bar_b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5, color="darkorange")
            if delta is not None:
                color = "green" if delta >= 0 else "red"
                ax.text(bar_p.get_x() + bar_p.get_width(),
                        max(bar_p.get_height(), bar_b.get_height()) + 0.04,
                        f"Δ{delta:+.3f}", ha="center", va="bottom", fontsize=7, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(task_names, fontsize=8)
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1.0)
        ax.set_title("SAE Probe vs Baseline: Macro F1 on Validation Set\n(Probe trained on 100% train, evaluated on held-out validation)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / "comparison_probe_vs_baseline.png", dpi=150)
        plt.close(fig)
        logger.info("Saved comparison_probe_vs_baseline.png")

    # ── 2. Per-class recall heatmaps (probe vs baseline side by side) ─────────
    for result in comparison_rows:
        if result["baseline_metrics"] is None:
            continue
        probe_pc = result["probe_metrics"].get("per_class", {})
        base_pc = result["baseline_metrics"].get("per_class", {})
        all_labels = sorted(set(probe_pc) | set(base_pc))
        if len(all_labels) < 2:
            continue

        recall_probe = np.array([[probe_pc.get(lbl, {}).get("recall", 0) for lbl in all_labels]])
        recall_base = np.array([[base_pc.get(lbl, {}).get("recall", 0) for lbl in all_labels]])
        mat = np.vstack([recall_probe, recall_base])

        fig, ax = plt.subplots(figsize=(max(5, len(all_labels) * 1.4), 3))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Probe", "Baseline"], fontsize=9)

        for j, lbl in enumerate(all_labels):
            for row_idx, pc in enumerate([probe_pc, base_pc]):
                v = pc.get(lbl, {})
                r_val = v.get("recall", 0)
                f_val = v.get("f1", 0)
                ax.text(j, row_idx, f"R={r_val:.2f}\nF1={f_val:.2f}",
                        ha="center", va="center", fontsize=7, color="black")

        safe = result["task_key"].replace("::", "__").replace(" ", "_").replace("[", "").replace("]", "")
        ax.set_title(f"Recall: Probe vs Baseline\n{result['task_key']}", fontsize=9)
        fig.colorbar(im, ax=ax, label="Recall", shrink=0.8)
        fig.tight_layout()
        fig.savefig(output_dir / f"comparison_perclass_{safe}.png", dpi=150)
        plt.close(fig)

    logger.info("Plots saved to %s", output_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(
        ProbeValidateConfig,
        argument_generation_mode=ArgumentGenerationMode.FLAT,
    )
    run_validate(cfg)


if __name__ == "__main__":
    main()
