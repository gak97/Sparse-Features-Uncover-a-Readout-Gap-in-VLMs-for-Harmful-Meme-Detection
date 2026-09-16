

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)

# The latents to analyse — in-workspace routing features vs silent probe features
ROUTING_FEATURES = [6786, 5536, 5449, 5353]  # top J-space, not in probe top-200
SILENT_FEATURES = [10445, 15987, 10168, 10962, 9744]  # top probe weight, J-frac≈0


@dataclass
class JlensKeyFeaturesConfig:
    # Feature matrix from probe_fhm_crosscoder_65k_prompt_image_save_matrix
    feature_matrix_dir: str = ""  # dir containing feature_matrix.npz + feature_matrix_keys.npy

    # Optional: manifest for the dense validation cache (to get meme text, gold label)
    dense_cache_manifest: str = ""

    # Features to extract exemplars for (routing + silent)
    routing_feature_indices: list[int] = field(default_factory=lambda: ROUTING_FEATURES)
    silent_feature_indices: list[int] = field(default_factory=lambda: SILENT_FEATURES)

    top_k_exemplars: int = 20  # top activating examples per feature

    output_dir: str = ""
    make_plots: bool = True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalise_id(sid: str) -> str:
    return str(sid).split("/")[-1].split(".")[0]


def _load_feature_matrix(
    matrix_dir: Path,
) -> tuple[np.ndarray, list[str]]:
    """Load sparse CSR matrix and row keys. Returns (dense_matrix [n, n_latents], sample_ids)."""
    import scipy.sparse as sp

    keys_path = matrix_dir / "feature_matrix_keys.npy"
    matrix_path = matrix_dir / "feature_matrix.npz"
    assert keys_path.exists(), f"Missing {keys_path}"
    assert matrix_path.exists(), f"Missing {matrix_path}"

    keys = np.load(str(keys_path), allow_pickle=True).tolist()
    mat = sp.load_npz(str(matrix_path))
    logger.info(
        "Loaded feature matrix: shape=%s  nnz=%d  n_keys=%d",
        mat.shape, mat.nnz, len(keys),
    )
    return mat, keys  # type: ignore[return-value]


def _get_exemplars(
    mat: Any,  # sparse or array [n, n_latents]
    sample_ids: list[str],
    feature_idx: int,
    top_k: int,
    manifest_by_id: dict[str, dict],
) -> list[dict[str, Any]]:
    """Top-k highest-activation examples for a single feature."""
    import scipy.sparse as sp

    if sp.issparse(mat):
        col = np.asarray(mat.getcol(feature_idx).todense()).ravel()
    else:
        col = mat[:, feature_idx]

    top_indices = np.argsort(col)[::-1][:top_k]
    exemplars = []
    for idx in top_indices:
        activation = float(col[idx])
        if activation <= 0:
            break
        sid = _normalise_id(str(sample_ids[idx]))
        row = manifest_by_id.get(sid, {})
        exemplars.append({
            "sample_id": sid,
            "activation": activation,
            "meme_text": row.get("meme_text", ""),
            "gold_label": row.get("gold_fields", {}).get("label", row.get("gold_label", "")),
            "dataset_name": row.get("dataset_name", ""),
            "generated_text": row.get("generated_text", ""),
        })
    return exemplars


def _feature_stats(
    mat: Any,
    sample_ids: list[str],
    feature_indices: list[int],
    manifest_by_id: dict[str, dict],
) -> dict[int, dict[str, Any]]:
    """Activation statistics for each feature."""
    import scipy.sparse as sp

    stats: dict[int, dict[str, Any]] = {}
    for feat_idx in feature_indices:
        if feat_idx >= mat.shape[1]:
            logger.warning("Feature %d out of range (n_latents=%d)", feat_idx, mat.shape[1])
            continue

        if sp.issparse(mat):
            col = np.asarray(mat.getcol(feat_idx).todense()).ravel()
        else:
            col = mat[:, feat_idx]

        active = col[col > 0]
        # Gold label distribution of top-20 activating samples
        top_indices = np.argsort(col)[::-1][:20]
        top_labels = []
        for idx in top_indices:
            if col[idx] <= 0:
                break
            sid = _normalise_id(str(sample_ids[idx]))
            row = manifest_by_id.get(sid, {})
            lbl = row.get("gold_fields", {}).get("label", row.get("gold_label", "unknown"))
            top_labels.append(str(lbl))

        hateful_frac = top_labels.count("hateful") / len(top_labels) if top_labels else 0.0

        stats[feat_idx] = {
            "n_active": int(len(active)),
            "frac_active": float(len(active) / len(col)),
            "mean_active": float(active.mean()) if len(active) else 0.0,
            "max": float(col.max()),
            "top20_hateful_frac": hateful_frac,
            "top20_labels": top_labels,
        }
    return stats


def _plot_histograms(
    mat: Any,
    routing_indices: list[int],
    silent_indices: list[int],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scipy.sparse as sp
    except ImportError:
        return

    n_feats = len(routing_indices) + len(silent_indices)
    fig, axes = plt.subplots(2, max(len(routing_indices), len(silent_indices)),
                             figsize=(4 * max(len(routing_indices), len(silent_indices)), 5),
                             squeeze=False)

    for row_i, (label, indices, color) in enumerate([
        ("in-workspace routing", routing_indices, "steelblue"),
        ("silent probe", silent_indices, "firebrick"),
    ]):
        for col_i, feat_idx in enumerate(indices):
            ax = axes[row_i][col_i]
            if feat_idx < mat.shape[1]:
                if sp.issparse(mat):
                    col = np.asarray(mat.getcol(feat_idx).todense()).ravel()
                else:
                    col = mat[:, feat_idx]
                active = col[col > 0]
                if len(active):
                    ax.hist(active, bins=30, color=color, alpha=0.7)
                ax.set_title(f"f{feat_idx}\n({label})", fontsize=7)
                ax.set_xlabel("activation", fontsize=7)
            else:
                ax.set_visible(False)

    fig.suptitle("Activation distributions: routing (blue) vs silent probe (red) features", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _run(cfg: JlensKeyFeaturesConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_dir = Path(cfg.feature_matrix_dir)
    mat, sample_keys = _load_feature_matrix(matrix_dir)

    # Load manifest for metadata lookup
    manifest_by_id: dict[str, dict] = {}
    if cfg.dense_cache_manifest:
        manifest_path = Path(cfg.dense_cache_manifest)
        if not manifest_path.name.endswith(".jsonl"):
            manifest_path = manifest_path / "manifest.jsonl"
        if manifest_path.exists():
            for row in _read_jsonl(manifest_path):
                sid = _normalise_id(str(row.get("sample_id", "")))
                manifest_by_id[sid] = row
            logger.info("Loaded %d manifest rows", len(manifest_by_id))

    all_indices = cfg.routing_feature_indices + cfg.silent_feature_indices

    # Extract exemplars for all key features
    exemplars: dict[str, list[dict]] = {}
    for feat_idx in all_indices:
        if feat_idx >= mat.shape[1]:
            logger.warning("Feature %d out of range", feat_idx)
            continue
        group = "routing" if feat_idx in cfg.routing_feature_indices else "silent"
        exs = _get_exemplars(mat, sample_keys, feat_idx, cfg.top_k_exemplars, manifest_by_id)
        exemplars[str(feat_idx)] = exs
        logger.info(
            "f%d (%s): %d exemplars, max_act=%.4f, top5_labels=%s",
            feat_idx, group, len(exs),
            exs[0]["activation"] if exs else 0.0,
            [e["gold_label"] for e in exs[:5]],
        )

    (out_dir / "key_feature_exemplars.json").write_text(json.dumps(exemplars, indent=2))

    # Activation statistics
    routing_stats = _feature_stats(mat, sample_keys, cfg.routing_feature_indices, manifest_by_id)
    silent_stats = _feature_stats(mat, sample_keys, cfg.silent_feature_indices, manifest_by_id)

    comparison: dict[str, Any] = {
        "routing_features": {
            str(k): {"jspace_role": "in-workspace output-routing", **v}
            for k, v in routing_stats.items()
        },
        "silent_features": {
            str(k): {"jspace_role": "silent-knowledge probe-discriminative", **v}
            for k, v in silent_stats.items()
        },
    }
    (out_dir / "key_feature_comparison.json").write_text(json.dumps(comparison, indent=2))

    # Print comparison table
    print("\n=== Routing (in-workspace) features ===")
    for feat_idx, s in routing_stats.items():
        print(
            f"  f{feat_idx}: frac_active={s['frac_active']:.3f}  "
            f"mean_act={s['mean_active']:.4f}  max={s['max']:.4f}  "
            f"top20_hateful={s['top20_hateful_frac']:.2f}"
        )

    print("\n=== Silent (probe-discriminative) features ===")
    for feat_idx, s in silent_stats.items():
        print(
            f"  f{feat_idx}: frac_active={s['frac_active']:.3f}  "
            f"mean_act={s['mean_active']:.4f}  max={s['max']:.4f}  "
            f"top20_hateful={s['top20_hateful_frac']:.2f}"
        )

    if cfg.make_plots:
        _plot_histograms(
            mat,
            cfg.routing_feature_indices,
            cfg.silent_feature_indices,
            out_dir / "activation_histograms.png",
        )

    logger.info("Done — outputs written to %s", out_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(JlensKeyFeaturesConfig, argument_generation_mode=ArgumentGenerationMode.FLAT)
    _run(cfg)


if __name__ == "__main__":
    main()
