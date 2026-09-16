

import json
import logging
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from safetensors import safe_open
from simple_parsing import ArgumentGenerationMode, Serializable, list_field, parse

from gemma3_experiments.residual_sae import (
    DenseActivationCacheConfig,
    GemmaResidualConfig,
    MemeDatasetConfig,
    ROLE_GENERATED,
    ROLE_IMAGE,
    ROLE_PROMPT,
    run_dense_activation_cache,
)
from gemma3_experiments.residual_sae.gemma import GemmaSae

logger = logging.getLogger(__name__)

_SAE_CHUNK = 256


@dataclass
class PilotQualityConfig(Serializable):
    dataset: MemeDatasetConfig = field(default_factory=lambda: MemeDatasetConfig(sample_limit=2000))
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    output_dir: str = "./pilot_quality_report"
    cache_dir: str = "./pilot_quality_cache"
    selections: list[str] = list_field("all", "image_and_generated", "image")
    overwrite: bool = True
    two_pass_fvu: bool = True


def _selection_mask(token_roles: torch.Tensor, selection: str) -> torch.Tensor:
    if selection == "all":
        return torch.ones_like(token_roles, dtype=torch.bool)
    if selection == "image":
        return token_roles == ROLE_IMAGE
    if selection == "image_and_generated":
        return (token_roles == ROLE_IMAGE) | (token_roles == ROLE_GENERATED)
    raise ValueError(f"Unsupported selection mode: {selection!r}")


def _analyze_selection(
    shard_paths: list[Path],
    base_sae: GemmaSae,
    selection: str,
    sae_device: str,
    two_pass_fvu: bool,
) -> dict:
    """Compute quality stats for the given token selection mode across all shards."""
    n_prompt = n_image = n_generated = 0
    n_selected = 0
    n_nan = n_inf = 0
    hidden_sum_sq = 0.0  # sum of ||h||^2 over selected tokens
    residual_sum_sq = 0.0  # sum of ||h - recon||^2

    # First pass: compute sums for all stats + (optionally) global mean for true FVU.
    global_mean: torch.Tensor | None = None
    global_dim: int | None = None

    for shard_path in shard_paths:
        with safe_open(str(shard_path), framework="pt", device="cpu") as fd:
            hidden = fd.get_tensor("hidden")          # (n_tok, d)
            token_roles = fd.get_tensor("token_roles")  # (n_tok,)

        # Accumulate role counts (over ALL tokens in shard, not just selected).
        n_prompt += int((token_roles == ROLE_PROMPT).sum())
        n_image += int((token_roles == ROLE_IMAGE).sum())
        n_generated += int((token_roles == ROLE_GENERATED).sum())

        mask = _selection_mask(token_roles, selection)
        sel = hidden[mask]
        if sel.shape[0] == 0:
            continue

        n_selected += int(sel.shape[0])

        if not torch.isfinite(sel).all():
            n_nan += int(torch.isnan(sel).any())
            n_inf += int(torch.isinf(sel).any())

        if global_dim is None:
            global_dim = int(sel.shape[1])
        if two_pass_fvu:
            if global_mean is None:
                global_mean = sel.to(torch.float64).sum(0)
            else:
                global_mean += sel.to(torch.float64).sum(0)

        # Compute SAE reconstruction in chunks and accumulate squared norms.
        for start in range(0, sel.shape[0], _SAE_CHUNK):
            chunk = sel[start : start + _SAE_CHUNK].to(sae_device, dtype=torch.float32)
            with torch.no_grad():
                _, recon = base_sae.reconstruct(chunk)
                residual = chunk - recon
            hidden_sum_sq += float(chunk.pow(2).sum().item())
            residual_sum_sq += float(residual.pow(2).sum().item())
            del chunk, recon, residual

    # Finalise first-pass aggregates.
    if n_selected == 0:
        return {
            "n_selected": 0,
            "n_prompt": n_prompt,
            "n_image": n_image,
            "n_generated": n_generated,
            "has_nan": bool(n_nan > 0),
            "has_inf": bool(n_inf > 0),
            "mean_hidden_l2": None,
            "mean_residual_l2": None,
            "base_sae_fvu_proxy": None,
            "base_sae_true_fvu": None,
        }

    mean_hidden_sq = hidden_sum_sq / n_selected
    mean_residual_sq = residual_sum_sq / n_selected
    fvu_proxy = mean_residual_sq / max(mean_hidden_sq, 1e-12)

    true_fvu: float | None = None
    if two_pass_fvu and global_mean is not None:
        global_mean = (global_mean / n_selected).to(torch.float32)  # (d,)

        # Second pass: compute E[||h - mean(h)||^2].
        total_variance_sq = 0.0
        for shard_path in shard_paths:
            with safe_open(str(shard_path), framework="pt", device="cpu") as fd:
                hidden = fd.get_tensor("hidden")
                token_roles = fd.get_tensor("token_roles")
            mask = _selection_mask(token_roles, selection)
            sel = hidden[mask]
            if sel.shape[0] == 0:
                continue
            for start in range(0, sel.shape[0], _SAE_CHUNK):
                chunk = sel[start : start + _SAE_CHUNK].to(sae_device, dtype=torch.float32)
                centered = chunk - global_mean.to(sae_device)
                total_variance_sq += float(centered.pow(2).sum().item())
                del chunk, centered

        mean_total_var_sq = total_variance_sq / n_selected
        true_fvu = mean_residual_sq / max(mean_total_var_sq, 1e-12)

    return {
        "n_selected": n_selected,
        "n_prompt": n_prompt,
        "n_image": n_image,
        "n_generated": n_generated,
        "has_nan": bool(n_nan > 0),
        "has_inf": bool(n_inf > 0),
        "mean_hidden_l2": round(math.sqrt(mean_hidden_sq), 4),
        "mean_residual_l2": round(math.sqrt(mean_residual_sq), 4),
        "base_sae_fvu_proxy": round(fvu_proxy, 4),
        "base_sae_true_fvu": round(true_fvu, 4) if true_fvu is not None else None,
    }


def _print_table(report: dict, output_dir: Path) -> None:
    selections = list(report.get("selections", {}).keys())
    if not selections:
        return

    cols = [
        ("selection", 22),
        ("n_selected", 12),
        ("n_image", 10),
        ("n_gen", 10),
        ("has_nan", 9),
        ("has_inf", 9),
        ("hidden_l2", 11),
        ("resid_l2", 10),
        ("fvu_proxy", 11),
        ("true_fvu", 10),
    ]
    header = "  ".join(f"{name:<{w}}" for name, w in cols)
    sep = "  ".join("-" * w for _, w in cols)
    lines = [header, sep]
    for sel in selections:
        s = report["selections"][sel]
        row_vals = [
            sel,
            str(s.get("n_selected", "-")),
            str(s.get("n_image", "-")),
            str(s.get("n_generated", "-")),
            "YES" if s.get("has_nan") else "no",
            "YES" if s.get("has_inf") else "no",
            str(s.get("mean_hidden_l2", "-")),
            str(s.get("mean_residual_l2", "-")),
            str(s.get("base_sae_fvu_proxy", "-")),
            str(s.get("base_sae_true_fvu", "-")),
        ]
        lines.append("  ".join(f"{v:<{w}}" for v, (_, w) in zip(row_vals, cols)))

    table = "\n".join(lines)
    print("\n=== Pilot Cache Quality Report ===")
    print(f"  Cache dir : {report.get('cache_dir', '?')}")
    print(f"  Samples   : {report.get('n_samples', '?')}")
    print(f"  SAE       : {report.get('sae_hookpoint', '?')}")
    print()
    print(table)
    print()

    txt_fpath = output_dir / "pilot_quality_summary.txt"
    with txt_fpath.open("w") as fd:
        fd.write("=== Pilot Cache Quality Report ===\n")
        fd.write(f"cache_dir  : {report.get('cache_dir', '?')}\n")
        fd.write(f"n_samples  : {report.get('n_samples', '?')}\n")
        fd.write(f"sae        : {report.get('sae_hookpoint', '?')}\n\n")
        fd.write(table + "\n")
    logger.info("Summary written to %s", txt_fpath)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(PilotQualityConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.cache_dir)

    assert cfg.dataset.sample_limit is not None and cfg.dataset.sample_limit > 0, (
        "Set --dataset.sample_limit (or --sample_limit) to a positive integer (e.g. 2000) for the pilot."
    )

    logger.info(
        "Pilot quality evaluation: sample_limit=%d, selections=%s, two_pass_fvu=%s",
        cfg.dataset.sample_limit,
        cfg.selections,
        cfg.two_pass_fvu,
    )

    # 1. Build dense cache with selection="all" (preserves all token roles for later filtering).
    dense_cfg = DenseActivationCacheConfig(
        dataset=cfg.dataset,
        gemma=replace(cfg.gemma, token_selection="all", capture_generation=True),
        output_dir=str(cache_dir),
        overwrite=cfg.overwrite,
        resume=False,
    )
    logger.info("Building dense activation cache at %s …", cache_dir)
    t0 = time.monotonic()
    cache_meta = run_dense_activation_cache(dense_cfg)
    logger.info("Dense cache complete in %.0fs.", time.monotonic() - t0)

    # 2. Load the base Gemma-Scope SAE for FVU computation.
    logger.info("Loading base SAE %s / %s on %s …", cfg.gemma.sae_repo_id, cfg.gemma.sae_hookpoint, cfg.gemma.sae_device)
    base_sae = GemmaSae.load_from_hub(
        cfg.gemma.sae_repo_id,
        cfg.gemma.sae_hookpoint,
        device=cfg.gemma.sae_device,
    )
    base_sae.eval()
    logger.info("Base SAE loaded: d_in=%d, num_latents=%d", base_sae.d_in, base_sae.num_latents)

    # 3. Gather shard paths.
    shard_paths = sorted(cache_dir.glob("dense_*.safetensors"))
    assert shard_paths, f"No dense shards found in {cache_dir}"
    logger.info("Found %d dense shards.", len(shard_paths))

    # 4. Analyse each selection mode.
    report: dict = {
        "cache_dir": str(cache_dir),
        "n_shards": len(shard_paths),
        "n_samples": cache_meta.get("n_samples", "?"),
        "sae_repo_id": cfg.gemma.sae_repo_id,
        "sae_hookpoint": cfg.gemma.sae_hookpoint,
        "two_pass_fvu": cfg.two_pass_fvu,
        "selections": {},
    }

    for selection in cfg.selections:
        logger.info("Analysing selection=%s …", selection)
        t1 = time.monotonic()
        stats = _analyze_selection(shard_paths, base_sae, selection, cfg.gemma.sae_device, cfg.two_pass_fvu)
        elapsed = time.monotonic() - t1
        report["selections"][selection] = stats
        logger.info(
            "selection=%-22s  n_selected=%7d  hidden_l2=%s  resid_l2=%s  "
            "fvu_proxy=%s  true_fvu=%s  nan=%s  inf=%s  (%.0fs)",
            selection,
            stats.get("n_selected", 0),
            stats.get("mean_hidden_l2", "-"),
            stats.get("mean_residual_l2", "-"),
            stats.get("base_sae_fvu_proxy", "-"),
            stats.get("base_sae_true_fvu", "-"),
            stats.get("has_nan"),
            stats.get("has_inf"),
            elapsed,
        )

    # 5. Save report.
    report_fpath = output_dir / "pilot_quality_report.json"
    with report_fpath.open("w") as fd:
        json.dump(report, fd, indent=2)
    logger.info("Report saved to %s", report_fpath)

    _print_table(report, output_dir)


if __name__ == "__main__":
    main()
