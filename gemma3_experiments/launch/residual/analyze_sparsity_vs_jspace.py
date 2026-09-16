

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)

# Features of special interest for labelling in plots
ROUTING_FEATURES = {6786: "f6786 (yes/no router)", 5536: "f5536 (hate)", 5449: "f5449 (broad)", 5353: "f5353 (offensive)"}
SILENT_FEATURES = {10445: "f10445 (probe#1)", 15987: "f15987 (probe#2)", 10168: "f10168 (never fires)"}


@dataclass
class SparsityVsJspaceConfig:
    # Feature matrix (validation split, sparse CSR)
    feature_matrix_dir: str = ""  # dir with feature_matrix.npz + feature_matrix_keys.npy

    # J-lens analysis dir (from analyze_jacobian_lens_features.py)
    # Must contain jlens_all_latents_top_inworkspace.json
    jlens_analysis_dir: str = ""

    # Optional: conditional J-lens dir for f15987 active samples
    conditional_jlens_analysis_dir: str = ""
    conditional_feature_idx: int = 15987

    output_dir: str = ""
    make_plots: bool = True


def _run(cfg: SparsityVsJspaceConfig) -> None:
    import scipy.sparse as sp

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load feature matrix ──────────────────────────────────────────────────
    matrix_dir = Path(cfg.feature_matrix_dir)
    mat = sp.load_npz(str(matrix_dir / "feature_matrix.npz"))
    n_samples, n_latents = mat.shape
    logger.info("Feature matrix: %d samples × %d latents", n_samples, n_latents)

    # Activation frequency per latent
    frac_active = np.diff(mat.tocsc().indptr) / n_samples  # [n_latents]
    logger.info(
        "frac_active: min=%.4f  median=%.4f  max=%.4f  n_tonic(>0.99)=%d",
        frac_active.min(), float(np.median(frac_active)), frac_active.max(),
        int((frac_active > 0.99).sum()),
    )

    # ── Load J-space fractions ───────────────────────────────────────────────
    jlens_dir = Path(cfg.jlens_analysis_dir)
    jlens_data = json.loads((jlens_dir / "jlens_all_latents_top_inworkspace.json").read_text())
    top_inws = jlens_data["top_100_inworkspace"]


    # Find the J-lens safetensors (the J-matrix, not the analysis dir)
    # jlens_analysis_dir is e.g. .../jlens_analysis_crosscoder_4b_l22
    # The matrix itself is in .../jacobian_lens_4b_fhm/jlens_matrix_layer22.safetensors
    # Try to find the J-matrix by looking for it in the parent dir
    jlens_matrix_path: Path | None = None
    for candidate in [
        jlens_dir.parent / "jacobian_lens_4b_fhm" / "jlens_matrix_layer22.safetensors",
        jlens_dir.parent / "jacobian_lens_4b_fhm" / "jlens_matrix_layer29.safetensors",
        jlens_dir.parent / "jlens_matrix_layer22.safetensors",
    ]:
        if candidate.exists():
            jlens_matrix_path = candidate
            break

    # Also try to load crosscoder w_dec for full projection
    # We work with the top-100 j_space fractions from the existing JSON
    # and per-feature frac_active to build the scatter
    # For a full scatter we need the jspace fractions for all 16k features — compute below.

    jspace_frac: np.ndarray | None = None
    if jlens_matrix_path is not None:
        try:
            from safetensors.torch import load_file as safe_load
            import torch
            j_avg = safe_load(str(jlens_matrix_path))
            j_avg_tensor = next(iter(j_avg.values())).float()  # [n_target, d_model]
            logger.info("Loaded J-avg matrix: shape=%s", tuple(j_avg_tensor.shape))

            # Load crosscoder w_dec
            from huggingface_hub import snapshot_download
            repo_path = (
                Path(snapshot_download(
                    "google/gemma-scope-2-4b-it",
                    allow_patterns=["crosscoder/layer_9_17_22_29_width_65k_l0_medium/*.safetensors"],
                ))
                / "crosscoder/layer_9_17_22_29_width_65k_l0_medium"
            )
            st_paths = sorted(repo_path.rglob("*.safetensors"))
            params_path = next((p for p in st_paths if p.name == "params.safetensors"), st_paths[0])
            sd = safe_load(str(params_path), device="cpu")
            w_dec = sd["w_dec"].float()[:, 2, :]  # layer-22 slice: [n_latents, d_model]
            logger.info("Loaded crosscoder w_dec: shape=%s", tuple(w_dec.shape))

            # Compute J-space fractions for all latents
            all_proj = j_avg_tensor @ w_dec.T  # [n_target, n_latents]
            all_proj_norms = all_proj.norm(dim=0).numpy()  # [n_latents]
            w_dec_norms = w_dec.norm(dim=-1).numpy()      # [n_latents]
            jspace_frac = all_proj_norms / (w_dec_norms + 1e-10)  # [n_latents]
            logger.info(
                "J-space fractions: min=%.6f  median=%.6f  max=%.6f",
                float(jspace_frac.min()), float(np.median(jspace_frac)), float(jspace_frac.max()),
            )
        except Exception as exc:
            logger.warning("Could not compute full J-space fractions: %s — using top-100 only", exc)

    # Fall back to top-100 if full computation failed
    if jspace_frac is None:
        jspace_frac = np.zeros(n_latents)
        for item in top_inws:
            fi = int(item["feature_idx"])
            if fi < n_latents:
                jspace_frac[fi] = float(item["jspace_fraction"])

    # ── Correlation and binning ──────────────────────────────────────────────
    pearson_r = float(np.corrcoef(frac_active, jspace_frac)[0, 1])
    logger.info("Pearson r(frac_active, jspace_fraction) = %.4f", pearson_r)

    # Bin by frac_active quartiles
    q_edges = [0.0, 0.01, 0.10, 0.50, 1.01]
    bin_labels = ["0–1%", "1–10%", "10–50%", "50–100%"]
    bin_stats: list[dict] = []
    for lo, hi, label in zip(q_edges[:-1], q_edges[1:], bin_labels):
        mask = (frac_active >= lo) & (frac_active < hi)
        n_in = int(mask.sum())
        mean_jfrac = float(jspace_frac[mask].mean()) if n_in else 0.0
        median_jfrac = float(np.median(jspace_frac[mask])) if n_in else 0.0
        bin_stats.append({
            "bin": label,
            "n_latents": n_in,
            "mean_jspace_fraction": mean_jfrac,
            "median_jspace_fraction": median_jfrac,
        })
        logger.info(
            "Bin %s: n=%d  mean_jfrac=%.6f  median_jfrac=%.6f",
            label, n_in, mean_jfrac, median_jfrac,
        )

    # Key feature stats
    key_features: dict[str, Any] = {}
    for feat_idx, label in {**ROUTING_FEATURES, **SILENT_FEATURES}.items():
        if feat_idx < n_latents:
            key_features[str(feat_idx)] = {
                "label": label,
                "frac_active": float(frac_active[feat_idx]),
                "jspace_fraction": float(jspace_frac[feat_idx]),
                "group": "routing" if feat_idx in ROUTING_FEATURES else "silent",
            }

    # ── Conditional J-lens comparison ────────────────────────────────────────
    conditional_comparison: dict[str, Any] = {}
    if cfg.conditional_jlens_analysis_dir:
        cond_dir = Path(cfg.conditional_jlens_analysis_dir)
        cond_path = cond_dir / "jlens_all_latents_top_inworkspace.json"
        if cond_path.exists():
            cond_data = json.loads(cond_path.read_text())
            cond_top = {int(r["feature_idx"]): float(r["jspace_fraction"]) for r in cond_data["top_100_inworkspace"]}
            # Compare conditional vs unconditional for our key features
            feat_idx = cfg.conditional_feature_idx
            unconditional = float(jspace_frac[feat_idx]) if feat_idx < n_latents else None
            conditional = cond_top.get(feat_idx)
            conditional_comparison = {
                "feature_idx": feat_idx,
                "unconditional_jspace_fraction": unconditional,
                "conditional_jspace_fraction": conditional,
                "ratio_conditional_over_unconditional": (
                    conditional / unconditional if unconditional and unconditional > 0 else None
                ),
            }
            logger.info(
                "Conditional J-lens for f%d: unconditional=%.6f  conditional=%s  ratio=%s",
                feat_idx, unconditional or 0.0,
                f"{conditional:.6f}" if conditional else "not in top-100",
                f"{conditional_comparison['ratio_conditional_over_unconditional']:.1f}×"
                if conditional_comparison.get("ratio_conditional_over_unconditional") else "N/A",
            )

    # ── Write outputs ────────────────────────────────────────────────────────
    result: dict[str, Any] = {
        "n_samples": n_samples,
        "n_latents": n_latents,
        "pearson_r_frac_active_vs_jspace": pearson_r,
        "bin_stats": bin_stats,
        "key_features": key_features,
        "conditional_comparison": conditional_comparison,
        "n_tonic_features_gt99pct": int((frac_active > 0.99).sum()),
        "n_sparse_features_lt1pct": int((frac_active < 0.01).sum()),
    }
    (out_dir / "sparsity_vs_jspace.json").write_text(json.dumps(result, indent=2))

    if cfg.make_plots:
        _plot_scatter(frac_active, jspace_frac, key_features, pearson_r, out_dir / "scatter_sparsity_vs_jspace.png")
        _plot_bins(bin_stats, out_dir / "bin_jspace_by_sparsity.png")

    logger.info("Done — outputs written to %s", out_dir)


def _plot_scatter(
    frac_active: np.ndarray,
    jspace_frac: np.ndarray,
    key_features: dict,
    pearson_r: float,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    # All features (tiny dots)
    ax.scatter(frac_active, jspace_frac, s=1, alpha=0.15, color="steelblue", rasterized=True)

    # Highlight key features
    for feat_idx, info in key_features.items():
        fa = info["frac_active"]
        jf = info["jspace_fraction"]
        color = "red" if info["group"] == "silent" else "green"
        ax.scatter(fa, jf, s=60, color=color, zorder=5, edgecolors="black", lw=0.5)
        ax.annotate(info["label"], (fa, jf), fontsize=6, ha="left", va="bottom",
                    xytext=(3, 2), textcoords="offset points")

    ax.set_xlabel("Activation frequency (frac_active)")
    ax.set_ylabel("J-space fraction  ||J_avg @ w_dec|| / ||w_dec||")
    ax.set_title(f"Sparsity vs J-space visibility\n"
                 f"r={pearson_r:.3f}  (green=routing, red=silent probe features)")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_bins(bin_stats: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [b["bin"] for b in bin_stats]
    means = [b["mean_jspace_fraction"] for b in bin_stats]
    ns = [b["n_latents"] for b in bin_stats]
    x = range(len(labels))

    bars = ax.bar(x, means, color="steelblue", alpha=0.8)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"n={n:,}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Activation frequency bin")
    ax.set_ylabel("Mean J-space fraction")
    ax.set_title("Mean J-space fraction by feature activation frequency\n"
                 "(supports: sparse → invisible to averaged J-lens)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(SparsityVsJspaceConfig, argument_generation_mode=ArgumentGenerationMode.FLAT)
    _run(cfg)


if __name__ == "__main__":
    main()
