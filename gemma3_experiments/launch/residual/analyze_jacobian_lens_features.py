

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JacobianLensFeatureAnalysisConfig:
    # ── J-lens inputs ────────────────────────────────────────────────────────
    # Directory produced by compute_jacobian_lens.py
    jlens_dir: str = ""

    # Which layer's J-matrix to use (must match a file in jlens_dir)
    layer_idx: int = 31

    # ── Feature source (pick one or more) ────────────────────────────────────
    # Residual SAE checkpoint dir (contains sae.safetensors with w_dec key)
    sae_checkpoint_dir: str = ""

    # 4B/12B crosscoder from Gemma Scope
    crosscoder_repo_id: str = ""           # e.g. "google/gemma-scope-2-4b-it"
    crosscoder_hookpoint: str = ""         # e.g. "crosscoder/layer_9_17_22_29_width_65k_l0_medium"
    crosscoder_layer_idx: int = 2          # which layer slice to use

    # Bilinear factors: path to saved U/V matrices from probe_fhm_crosscoder_pairwise
    bilinear_checkpoint_dir: str = ""

    # ── Probe top features ───────────────────────────────────────────────────
    # top_features.json from probe_sae_features.py or probe_fhm_crosscoder.py
    probe_top_features_path: str = ""
    top_k_features: int = 200

    # ── Static logit alignment (for comparison) ──────────────────────────────
    # feature_logit_alignment.json from analyze_feature_logit_alignment.py
    static_logit_alignment_path: str = ""

    # ── Analysis thresholds ──────────────────────────────────────────────────
    # A feature is "high probe weight" if |probe_weight| > this percentile
    high_probe_percentile: float = 50.0
    # A feature is "in J-workspace" if ||J_avg @ w_dec[i]|| > this threshold
    # (normalised by ||w_dec[i]|| to give J-space fraction)
    jspace_fraction_threshold: float = 0.05
    # For static logit effect: same threshold applied to |static_effect|
    static_logit_threshold: float = 0.10

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = ""
    make_plots: bool = True
    # Target tokens to highlight in scatter plots
    highlight_tokens: list[str] = field(default_factory=lambda: [
        "hateful", "racist", "misogynistic", "safe", "benign",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_w_dec(cfg: JacobianLensFeatureAnalysisConfig) -> tuple[torch.Tensor, str]:
    """Return (w_dec [n_latents, d_model], source_label)."""
    from safetensors.torch import load_file as safe_load

    if cfg.sae_checkpoint_dir:
        ckpt = Path(cfg.sae_checkpoint_dir)
        sae_file = ckpt / "sae.safetensors"
        assert sae_file.exists(), f"Missing {sae_file}"
        sd = safe_load(str(sae_file), device="cpu")
        for key in ("w_dec", "W_dec", "decoder.weight"):
            if key in sd:
                logger.info("SAE w_dec: %s  shape=%s", key, tuple(sd[key].shape))
                return sd[key].float(), "residual_sae"
        raise KeyError(f"Cannot find w_dec in {sae_file}: keys={list(sd.keys())}")

    if cfg.crosscoder_repo_id and cfg.crosscoder_hookpoint:
        from huggingface_hub import snapshot_download
        repo_path = (
            Path(snapshot_download(
                cfg.crosscoder_repo_id,
                allow_patterns=[
                    f"{cfg.crosscoder_hookpoint}/*.safetensors",
                    f"{cfg.crosscoder_hookpoint}/*/*.safetensors",
                ],
            ))
            / cfg.crosscoder_hookpoint
        )
        safetensor_paths = sorted(repo_path.rglob("*.safetensors"))
        assert safetensor_paths, f"No safetensors found under {repo_path}"
        params_paths = [p for p in safetensor_paths if p.name == "params.safetensors"]
        load_path = params_paths[0] if params_paths else safetensor_paths[0]
        sd = safe_load(str(load_path), device="cpu")
        for key in ("w_dec", "W_dec", "decoder.weight"):
            if key in sd:
                w = sd[key].float()
                if w.ndim == 3:
                    w = w[:, cfg.crosscoder_layer_idx, :]
                logger.info("Crosscoder w_dec slice: shape=%s", tuple(w.shape))
                return w, "crosscoder"
        raise KeyError(f"Cannot find w_dec in {load_path}")

    if cfg.bilinear_checkpoint_dir:
        ckpt = Path(cfg.bilinear_checkpoint_dir)
        pt_file = ckpt / "model_checkpoint.pt"
        feat_idx_file = ckpt / "bilinear_feature_indices.json"
        metrics_file = ckpt / "pairwise_probe_metrics.json"

        if pt_file.exists() and feat_idx_file.exists() and metrics_file.exists():
            import torch as _torch
            feat_data = json.loads(feat_idx_file.read_text())
            feat_indices: list[int] = (
                feat_data["feature_indices"] if isinstance(feat_data, dict) else feat_data
            )
            logger.info("Bilinear: %d feature indices loaded from %s", len(feat_indices), feat_idx_file)

            # Parse crosscoder/SAE config from probe metrics
            metrics = json.loads(metrics_file.read_text())
            cfg_meta = metrics.get("config", {})
            cc_repo = cfg_meta.get("crosscoder_repo_id", "")
            cc_hookpoint = cfg_meta.get("crosscoder_hookpoint", "")
            assert cc_repo and cc_hookpoint, (
                f"Cannot find crosscoder_repo_id / crosscoder_hookpoint in {metrics_file}"
            )

            # Download and load the SAE/crosscoder w_dec
            from huggingface_hub import snapshot_download as _hf_dl
            repo_path = (
                Path(_hf_dl(
                    cc_repo,
                    allow_patterns=[
                        f"{cc_hookpoint}/*.safetensors",
                        f"{cc_hookpoint}/*/*.safetensors",
                    ],
                ))
                / cc_hookpoint
            )
            st_paths = sorted(repo_path.rglob("*.safetensors"))
            assert st_paths, f"No safetensors found under {repo_path}"
            load_path = next((p for p in st_paths if p.name == "params.safetensors"), st_paths[0])
            sd = safe_load(str(load_path), device="cpu")
            w_full: _torch.Tensor | None = None
            for key in ("w_dec", "W_dec", "decoder.weight"):
                if key in sd:
                    w_full = sd[key].float()
                    break
            assert w_full is not None, f"Cannot find w_dec in {load_path}"
            if w_full.ndim == 3:
                # multi-layer crosscoder: [n_latents, n_layers, d_model]; use first layer
                w_full = w_full[:, 0, :]
            logger.info("Loaded SAE w_dec from %s: shape=%s", cc_hookpoint, tuple(w_full.shape))

            # Load bilinear U factors for probe weights (row norms → feature importance)
            state = _torch.load(pt_file, map_location="cpu", weights_only=False)
            state_dict = state if isinstance(state, dict) and "bil_U" in state else (
                state.get("model_state_dict") or state
            )
            if "bil_U" in state_dict:
                bil_U = state_dict["bil_U"].float()  # [k, rank]
                probe_row_norms = bil_U.norm(dim=-1).tolist()  # [k]
            else:
                probe_row_norms = [1.0] * len(feat_indices)

            valid = [i for i in feat_indices if i < w_full.shape[0]]
            w_bilinear = w_full[valid]  # [k, d_model]
            logger.info(
                "Bilinear: %d / %d features valid, w_dec shape=%s",
                len(valid), len(feat_indices), tuple(w_bilinear.shape),
            )

            # Store row norms on the tensor as metadata (picked up by _run if present)
            w_bilinear._bilinear_probe_weights = probe_row_norms[:len(valid)]  # type: ignore[attr-defined]
            return w_bilinear, "bilinear_sae_features"

        # Fallback: look for old safetensors format
        u_file = ckpt / "U.safetensors"
        v_file = ckpt / "V.safetensors"
        if u_file.exists() and v_file.exists():
            U = safe_load(str(u_file), device="cpu")["U"].float()
            V = safe_load(str(v_file), device="cpu")["V"].float()
            w_dec = U.T  # [rank, d_model]
            logger.info("Bilinear U factors (safetensors): %d rank vectors × d_model=%d", *w_dec.shape)
            return w_dec, "bilinear_U"
        raise FileNotFoundError(
            f"Expected model_checkpoint.pt + bilinear_feature_indices.json in {ckpt}; "
            f"found: {[p.name for p in ckpt.iterdir() if p.is_file()]}"
        )

    raise ValueError(
        "Provide at least one of: --sae_checkpoint_dir, --crosscoder_repo_id, --bilinear_checkpoint_dir"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def _load_jlens_matrix(jlens_dir: Path, layer_idx: int) -> tuple[torch.Tensor, list[str]]:
    """Load averaged J-lens matrix and target token list."""
    from safetensors.torch import load_file as safe_load

    summary_path = jlens_dir / "jlens_summary.json"
    assert summary_path.exists(), f"Missing {summary_path}"
    summary = json.loads(summary_path.read_text())
    target_tokens = summary["target_tokens"]

    # Try safetensors first, then .pt
    st_path = jlens_dir / f"jlens_matrix_layer{layer_idx}.safetensors"
    pt_path = jlens_dir / f"jlens_matrix_layer{layer_idx}.pt"
    if st_path.exists():
        sd = safe_load(str(st_path), device="cpu")
        key = f"jlens_l{layer_idx}"
        j_avg = sd[key].float()
    elif pt_path.exists():
        j_avg = torch.load(pt_path, map_location="cpu").float()
    else:
        raise FileNotFoundError(
            f"No J-lens matrix found for layer {layer_idx} in {jlens_dir}"
        )

    logger.info(
        "Loaded J-lens matrix for layer %d: shape=%s  |J|_F=%.4f",
        layer_idx, tuple(j_avg.shape), float(j_avg.norm("fro")),
    )
    return j_avg, target_tokens


def _load_probe_top_features(path: str, top_k: int) -> dict[str, dict[str, list[dict]]]:
    """Load probe top features JSON.  Returns {task: {class: [{feature_idx, weight}]}}."""
    data = json.loads(Path(path).read_text())

    # Support two formats:
    # Format A (probe_sae_features): {task_key: {class: [{feature_idx, weight}]}}
    # Format B (probe_fhm_crosscoder): {class: [{feature_idx, weight}]} (flat, single task)
    if not data:
        return {}

    first_val = next(iter(data.values()))
    if isinstance(first_val, dict) and all(isinstance(v, list) for v in first_val.values()):
        # Format A
        result = {}
        for task_key, class_dict in data.items():
            result[task_key] = {
                cls: feats[:top_k] for cls, feats in class_dict.items()
            }
        return result

    if isinstance(first_val, list):
        # Format B — wrap under a synthetic task key
        return {"fhm_binary": {cls: feats[:top_k] for cls, feats in data.items()}}

    # Nested format with "rankings" key (from discover_features)
    logger.warning("Unrecognised top_features format; returning empty")
    return {}


def _compute_jlens_projections(
    j_avg: torch.Tensor,        # [n_target, d_model]
    w_dec: torch.Tensor,        # [n_latents, d_model]
    feature_indices: list[int],
) -> torch.Tensor:
    """
    Returns [len(feature_indices), n_target] J-lens projection scores.

    jlens_score[i, v] = J_avg[v] · w_dec[feature_idx_i]
    """
    w_sel = w_dec[feature_indices]  # [n_sel, d_model]
    # J_avg @ w_sel.T → [n_target, n_sel]; transpose → [n_sel, n_target]
    projections = (j_avg @ w_sel.T).T  # [n_sel, n_target]
    return projections.float()


def _jspace_fraction(
    j_avg: torch.Tensor,        # [n_target, d_model]
    w_dec_vec: torch.Tensor,    # [d_model]
) -> float:
    """||J_avg @ w_dec[i]|| / ||w_dec[i]|| — fraction of feature in J-workspace."""
    proj = j_avg @ w_dec_vec  # [n_target]
    return float(proj.norm() / (w_dec_vec.norm() + 1e-10))


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_probe_vs_jlens(
    probe_weights: list[float],
    jlens_fractions: list[float],
    feature_ids: list[int],
    task_label: str,
    class_label: str,
    out_path: Path,
    jspace_threshold: float,
    probe_threshold: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.array(jlens_fractions)
    ys = np.abs(np.array(probe_weights))

    ax.scatter(xs, ys, s=8, alpha=0.5, color="steelblue", label="features")

    # Quadrant lines
    ax.axvline(jspace_threshold, color="gray", lw=0.8, ls="--")
    ax.axhline(probe_threshold, color="gray", lw=0.8, ls="--")

    # Label quadrants
    ax.text(jspace_threshold * 0.1, probe_threshold * 1.05, "silent knowledge", fontsize=7, color="red")
    ax.text(jspace_threshold * 1.1, probe_threshold * 1.05, "aligned (in workspace)", fontsize=7, color="green")

    # Annotate top features
    top_n = 10
    order = np.argsort(-ys)[:top_n]
    for idx in order:
        ax.annotate(
            f"f{feature_ids[idx]}",
            (xs[idx], ys[idx]),
            fontsize=5,
            ha="center",
        )

    ax.set_xlabel("J-space fraction  ||J_avg @ w_dec|| / ||w_dec||")
    ax.set_ylabel("|probe weight|")
    ax.set_title(f"{task_label} / {class_label}\nJ-lens vs probe weight")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _scatter_static_vs_jlens(
    static_effects: list[float],
    jlens_fractions: list[float],
    feature_ids: list[int],
    task_label: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.array(jlens_fractions)
    ys = np.abs(np.array(static_effects))

    ax.scatter(xs, ys, s=8, alpha=0.4, color="darkorchid")
    ax.set_xlabel("J-space fraction  ||J_avg @ w_dec|| / ||w_dec||")
    ax.set_ylabel("|static logit effect|")
    ax.set_title(f"{task_label}\nStatic logit alignment vs J-lens projection")

    top_n = 10
    order = np.argsort(-ys)[:top_n]
    for idx in order:
        ax.annotate(f"f{feature_ids[idx]}", (xs[idx], ys[idx]), fontsize=5, ha="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _run(cfg: JacobianLensFeatureAnalysisConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jlens_dir = Path(cfg.jlens_dir)

    # ── Load J-lens matrix ───────────────────────────────────────────────────
    j_avg, target_tokens = _load_jlens_matrix(jlens_dir, cfg.layer_idx)
    # j_avg: [n_target, d_model]

    # ── Load feature decoder weights ─────────────────────────────────────────
    try:
        w_dec, source_label = _load_w_dec(cfg)
    except (FileNotFoundError, ValueError) as exc:
        msg = str(exc)
        logger.warning("Skipping analysis — cannot load feature weights: %s", msg)
        (out_dir / "jlens_feature_analysis_skipped.json").write_text(
            json.dumps({"skipped": True, "reason": msg}, indent=2)
        )
        return
    # w_dec: [n_latents, d_model]
    n_latents, d_model = w_dec.shape
    if d_model != j_avg.shape[1]:
        msg = (
            f"d_model mismatch: J-lens matrix has d_model={j_avg.shape[1]} "
            f"(from model_id used in compute_jacobian_lens.py) but w_dec has "
            f"d_model={d_model}. These must be from the same model family. "
            f"Run compute_jacobian_lens.py with the matching model to obtain a "
            f"compatible J-matrix."
        )
        logger.warning("Skipping analysis — %s", msg)
        (out_dir / "jlens_feature_analysis_skipped.json").write_text(
            json.dumps({"skipped": True, "reason": msg,
                        "jlens_d_model": j_avg.shape[1],
                        "w_dec_d_model": d_model}, indent=2)
        )
        return

    # ── Load probe top features (optional) ───────────────────────────────────
    probe_features: dict[str, dict[str, list[dict]]] = {}
    if cfg.probe_top_features_path:
        probe_features = _load_probe_top_features(cfg.probe_top_features_path, cfg.top_k_features)
        logger.info("Loaded probe top features for %d tasks", len(probe_features))
    elif hasattr(w_dec, "_bilinear_probe_weights"):
        # Auto-generate synthetic probe features from bilinear U row norms
        row_norms: list[float] = w_dec._bilinear_probe_weights  # type: ignore[attr-defined]
        feat_list_auto = [
            {"feature_idx": i, "weight": float(row_norms[i] if i < len(row_norms) else 1.0)}
            for i in range(n_latents)
        ]
        feat_list_auto.sort(key=lambda x: abs(x["weight"]), reverse=True)
        probe_features = {"bilinear": {"hateful": feat_list_auto[:cfg.top_k_features]}}
        logger.info(
            "Auto-generated %d bilinear probe features from U-factor row norms",
            len(feat_list_auto),
        )

    # ── Load static logit alignment (optional) ───────────────────────────────
    static_alignment: dict[str, Any] = {}
    if cfg.static_logit_alignment_path:
        static_alignment = json.loads(Path(cfg.static_logit_alignment_path).read_text())

    # ── J-lens scores for probe-top features ─────────────────────────────────
    jlens_results: dict[str, Any] = {}
    workspace_results: dict[str, Any] = {}

    for task_key, class_dict in probe_features.items():
        jlens_results[task_key] = {}
        workspace_results[task_key] = {}

        for cls_name, feat_list in class_dict.items():
            feat_indices = [int(f["feature_idx"]) for f in feat_list if int(f["feature_idx"]) < n_latents]
            probe_wts = {
                int(f["feature_idx"]): float(f.get("weight", f.get("score", 0.0)))
                for f in feat_list
            }

            if not feat_indices:
                continue

            # J-lens projections: [n_sel, n_target]
            projections = _compute_jlens_projections(j_avg, w_dec, feat_indices)

            feature_records = []
            for i, feat_idx in enumerate(feat_indices):
                w_vec = w_dec[feat_idx]
                jfrac = _jspace_fraction(j_avg, w_vec)
                proj_by_token = dict(zip(target_tokens, projections[i].tolist()))
                top_jlens_tokens = sorted(proj_by_token.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

                # Static logit alignment lookup
                static_effect = 0.0
                if static_alignment:
                    sa_task = static_alignment.get(task_key, {}).get(cls_name, {})
                    for frec in sa_task.get("features", []):
                        if int(frec.get("feature_idx", -1)) == feat_idx:
                            static_effect = float(frec.get("logit_effect", 0.0))
                            break

                feature_records.append({
                    "feature_idx": feat_idx,
                    "probe_weight": probe_wts.get(feat_idx, 0.0),
                    "jspace_fraction": jfrac,
                    "jlens_proj_by_token": proj_by_token,
                    "top_jlens_tokens": [{"token": t, "proj": v} for t, v in top_jlens_tokens],
                    "static_logit_effect": static_effect,
                })

            # Compute probe-weight threshold for this class
            abs_wts = [abs(r["probe_weight"]) for r in feature_records]
            probe_thresh = float(np.percentile(abs_wts, cfg.high_probe_percentile)) if abs_wts else 0.0

            # Classify into workspace quadrants
            in_workspace = []
            silent_knowledge = []
            static_only = []
            jlens_only = []
            for rec in feature_records:
                high_probe = abs(rec["probe_weight"]) >= probe_thresh
                in_jws = rec["jspace_fraction"] >= cfg.jspace_fraction_threshold
                high_static = abs(rec["static_logit_effect"]) >= cfg.static_logit_threshold

                if high_probe and in_jws:
                    in_workspace.append(rec["feature_idx"])
                elif high_probe and not in_jws and not high_static:
                    silent_knowledge.append(rec["feature_idx"])
                elif high_probe and high_static and not in_jws:
                    static_only.append(rec["feature_idx"])
                elif high_probe and in_jws and not high_static:
                    jlens_only.append(rec["feature_idx"])

            jlens_results[task_key][cls_name] = {
                "n_features": len(feature_records),
                "probe_weight_threshold": probe_thresh,
                "jspace_fraction_threshold": cfg.jspace_fraction_threshold,
                "features": feature_records,
            }
            workspace_results[task_key][cls_name] = {
                "in_workspace": in_workspace[:20],
                "silent_knowledge": silent_knowledge[:20],
                "static_only": static_only[:20],
                "jlens_only": jlens_only[:20],
                "counts": {
                    "in_workspace": len(in_workspace),
                    "silent_knowledge": len(silent_knowledge),
                    "static_only": len(static_only),
                    "jlens_only": len(jlens_only),
                    "total_high_probe": sum(1 for r in feature_records if abs(r["probe_weight"]) >= probe_thresh),
                },
            }

            logger.info(
                "Task=%s  class=%s  n=%d  in_workspace=%d  silent_knowledge=%d  static_only=%d",
                task_key, cls_name, len(feature_records),
                len(in_workspace), len(silent_knowledge), len(static_only),
            )

            # Scatter: probe weight vs J-space fraction
            if cfg.make_plots:
                _scatter_probe_vs_jlens(
                    probe_weights=[r["probe_weight"] for r in feature_records],
                    jlens_fractions=[r["jspace_fraction"] for r in feature_records],
                    feature_ids=[r["feature_idx"] for r in feature_records],
                    task_label=task_key,
                    class_label=cls_name,
                    out_path=out_dir / f"scatter_jlens_{task_key}_{cls_name}.png",
                    jspace_threshold=cfg.jspace_fraction_threshold,
                    probe_threshold=probe_thresh,
                )
                if static_alignment:
                    _scatter_static_vs_jlens(
                        static_effects=[r["static_logit_effect"] for r in feature_records],
                        jlens_fractions=[r["jspace_fraction"] for r in feature_records],
                        feature_ids=[r["feature_idx"] for r in feature_records],
                        task_label=f"{task_key}/{cls_name}",
                        out_path=out_dir / f"scatter_static_vs_jlens_{task_key}_{cls_name}.png",
                    )

    # ── Full-latent J-space fraction (no probe filter) ────────────────────────
    # Compute J-space fraction for ALL latents — can be used to rank which SAE
    # features are most "in the verbalizable workspace" regardless of probe.
    logger.info("Computing J-space fraction for all %d latents…", n_latents)
    # Batch matmul: J_avg @ w_dec.T → [n_target, n_latents]
    all_proj = j_avg @ w_dec.T  # [n_target, n_latents]
    all_proj_norms = all_proj.norm(dim=0)  # [n_latents]
    w_dec_norms = w_dec.norm(dim=-1)       # [n_latents]
    all_jfrac = (all_proj_norms / (w_dec_norms + 1e-10)).tolist()

    # Top-100 most in-workspace features
    top_inws_idx = np.argsort(all_jfrac)[::-1][:100].tolist()
    top_inws = [{"feature_idx": int(i), "jspace_fraction": float(all_jfrac[i])} for i in top_inws_idx]

    # Per-token top-20 features for each target token
    per_token_top: dict[str, list[dict]] = {}
    for ti, tok_str in enumerate(target_tokens):
        tok_proj = all_proj[ti, :]  # [n_latents]
        top_idx = tok_proj.abs().argsort(descending=True)[:20].tolist()
        per_token_top[tok_str] = [
            {"feature_idx": int(i), "proj": float(tok_proj[i])} for i in top_idx
        ]

    # ── Write outputs ─────────────────────────────────────────────────────────
    (out_dir / "jlens_feature_analysis.json").write_text(
        json.dumps(jlens_results, indent=2)
    )
    (out_dir / "jlens_workspace_features.json").write_text(
        json.dumps(workspace_results, indent=2)
    )
    (out_dir / "jlens_all_latents_top_inworkspace.json").write_text(
        json.dumps({
            "top_100_inworkspace": top_inws,
            "per_target_token_top20": per_token_top,
        }, indent=2)
    )

    # Cross-reference summary: how much does J-lens change the "silent" classification?
    if static_alignment:
        comparison: dict[str, Any] = {}
        for task_key in workspace_results:
            for cls_name, ws_data in workspace_results[task_key].items():
                sa_silent = set()
                sa_data = static_alignment.get(task_key, {}).get(cls_name, {})
                for frec in sa_data.get("features", []):
                    if frec.get("quadrant") == "silent":
                        sa_silent.add(int(frec["feature_idx"]))

                jlens_sk = set(ws_data["silent_knowledge"])
                comparison[f"{task_key}/{cls_name}"] = {
                    "static_logit_silent_count": len(sa_silent),
                    "jlens_silent_count": len(jlens_sk),
                    "reclassified_as_in_workspace_by_jlens": len(sa_silent - jlens_sk),
                    "newly_silent_by_jlens": len(jlens_sk - sa_silent),
                }
        (out_dir / "jlens_vs_static_comparison.json").write_text(
            json.dumps(comparison, indent=2)
        )

    logger.info(
        "Done. J-lens feature analysis written to %s  "
        "(source=%s, layer=%d, n_latents=%d)",
        out_dir, source_label, cfg.layer_idx, n_latents,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(
        JacobianLensFeatureAnalysisConfig,
        argument_generation_mode=ArgumentGenerationMode.FLAT,
    )
    _run(cfg)


if __name__ == "__main__":
    main()
