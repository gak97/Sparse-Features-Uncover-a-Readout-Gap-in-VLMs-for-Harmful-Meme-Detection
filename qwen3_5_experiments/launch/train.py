

import json
import logging
import math
import random
import time
import zlib
from pathlib import Path

import torch
from safetensors.torch import load_file
from simple_parsing import parse

from gemma3_experiments.residual_sae.normalization import ResidualStats, save_residual_stats

from ..config import ResidualSaeTrainConfig
from ..devices import resolve_visible_device
from ..models import JumpReluResidualSae

logger = logging.getLogger(__name__)


def _load_residual_shard_paths(cache_dir: Path) -> list[Path]:
    shard_paths = sorted(cache_dir.glob("residual_*.safetensors"))
    assert shard_paths, f"No residual shards found in {cache_dir}"
    return shard_paths


def _split_train_val_shards(
    shard_paths: list[Path],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    assert 0.0 <= val_fraction < 1.0, "val_fraction must be in [0, 1)"
    if val_fraction == 0.0 or len(shard_paths) < 2:
        return shard_paths, []
    rng = random.Random(seed)
    shuffled = list(shard_paths)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    n_val = min(n_val, len(shuffled) - 1)
    val_paths = sorted(shuffled[:n_val])
    train_paths = sorted(shuffled[n_val:])
    return train_paths, val_paths


def _iter_residual_batches(
    shard_paths: list[Path],
    *,
    batch_size: int,
    device: str,
    seed: int,
    epoch: int,
):
    rng = random.Random(seed + epoch)
    shuffled_paths = list(shard_paths)
    rng.shuffle(shuffled_paths)
    for shard_path in shuffled_paths:
        residual = load_file(str(shard_path))["residual"]
        shard_seed = zlib.crc32(shard_path.name.encode("utf-8"))
        generator = torch.Generator().manual_seed(seed + epoch * 1_000_003 + shard_seed)
        order = torch.randperm(residual.shape[0], generator=generator)
        residual = residual[order]
        for start in range(0, residual.shape[0], batch_size):
            batch = residual[start : start + batch_size]
            if batch.numel() == 0:
                continue
            yield batch.to(device=device, dtype=torch.float32), shard_path.name


def _compute_residual_stats(shard_paths: list[Path], eps: float) -> ResidualStats:
    d_in = int(load_file(str(shard_paths[0]))["residual"].shape[1])
    total = 0
    sum_x = torch.zeros(d_in, dtype=torch.float64)
    sum_x2 = torch.zeros(d_in, dtype=torch.float64)
    for shard_path in shard_paths:
        residual = load_file(str(shard_path))["residual"].to(dtype=torch.float64)
        total += int(residual.shape[0])
        sum_x += residual.sum(dim=0)
        sum_x2 += residual.square().sum(dim=0)
    assert total > 0, "Expected at least one residual token to compute normalization stats"
    mean = sum_x / total
    var = (sum_x2 / total) - mean.square()
    std = torch.sqrt(torch.clamp(var, min=eps**2))
    return ResidualStats(mean=mean.to(dtype=torch.float32), std=std.to(dtype=torch.float32), eps=eps)


def _normalize_batch(batch: torch.Tensor, stats: ResidualStats | None) -> torch.Tensor:
    return stats.normalize(batch) if stats is not None else batch


def _batch_total_variance(batch: torch.Tensor) -> float:
    return float((batch - batch.mean(0)).pow(2).sum().item())


def _initialize_decoder_bias(
    sae,
    shard_paths: list[Path],
    n_tokens: int,
    stats: ResidualStats | None,
) -> None:
    if n_tokens <= 0:
        return
    points: list[torch.Tensor] = []
    remaining = n_tokens
    for shard_path in shard_paths:
        residual = load_file(str(shard_path))["residual"]
        take = min(remaining, residual.shape[0])
        if take <= 0:
            break
        points.append(residual[:take])
        remaining -= take
        if remaining == 0:
            break
    if not points:
        return
    init_points = torch.cat(points, dim=0).to(device=sae.device, dtype=sae.dtype)
    init_points = _normalize_batch(init_points, stats)
    # Geometric median via Weiszfeld algorithm
    median = init_points.median(dim=0).values
    sae.b_dec.data.copy_(median.to(dtype=sae.b_dec.dtype, device=sae.b_dec.device))


def _parameter_stats(sae: JumpReluResidualSae) -> dict[str, float]:
    stats: dict[str, float] = {}
    with torch.no_grad():
        stats["b_dec_norm"] = float(sae.b_dec.norm().item())
        stats["b_dec_absmax"] = float(sae.b_dec.abs().max().item())
        stats["encoder_weight_norm"] = float(sae.encoder.weight.norm().item())
        stats["encoder_weight_absmax"] = float(sae.encoder.weight.abs().max().item())
        if hasattr(sae, "W_dec"):
            stats["W_dec_norm"] = float(sae.W_dec.norm().item())
            stats["W_dec_absmax"] = float(sae.W_dec.abs().max().item())
        if hasattr(sae, "threshold"):
            threshold = sae.threshold
            stats["threshold_mean"] = float(threshold.mean().item())
            stats["threshold_min"] = float(threshold.min().item())
            stats["threshold_max"] = float(threshold.max().item())
    return stats


def _latent_l0(out) -> float:
    if hasattr(out, "l0"):
        return float(out.l0.item())
    if out.latent_indices is not None:
        return float(out.latent_indices.shape[-1])
    return float((out.latent_acts > 0).sum(dim=-1).float().mean().item())


def _l0_loss(out, batch: torch.Tensor) -> torch.Tensor:
    if hasattr(out, "l0_loss"):
        return out.l0_loss
    return batch.new_tensor(0.0)


def _mark_fired(did_fire: torch.Tensor, out) -> None:
    if out.latent_indices is not None:
        did_fire[out.latent_indices.flatten()] = True
        return
    did_fire.logical_or_((out.latent_acts > 0).any(dim=0))


def _loss_from_output(out, batch: torch.Tensor, cfg: ResidualSaeTrainConfig) -> torch.Tensor:
    return (
        out.fvu
        + cfg.auxk_alpha * out.auxk_loss
        + cfg.multi_topk_loss_weight * out.multi_topk_fvu
        + _l0_loss(out, batch)
    )


def _set_train_mode(sae, train: bool) -> None:
    if train:
        sae.train()
    else:
        sae.eval()


def _evaluate_sae(
    sae,
    shard_paths: list[Path],
    *,
    batch_size: int,
    device: str,
    stats: ResidualStats | None,
    cfg: ResidualSaeTrainConfig,
) -> dict[str, float] | None:
    if not shard_paths:
        return None
    _set_train_mode(sae, False)
    loss_sum = fvu_sum = auxk_sum = multi_topk_sum = l0_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch, _shard_name in _iter_residual_batches(
            shard_paths,
            batch_size=batch_size,
            device=device,
            seed=cfg.seed + 17_171,
            epoch=0,
        ):
            batch = _normalize_batch(batch, stats)
            if batch.shape[0] < 2:
                continue
            out = sae(batch)
            loss = _loss_from_output(out, batch, cfg)
            loss_sum += float(loss.item())
            fvu_sum += float(out.fvu.item())
            auxk_sum += float(out.auxk_loss.item())
            multi_topk_sum += float(out.multi_topk_fvu.item())
            l0_sum += _latent_l0(out)
            n_batches += 1
            if cfg.eval_max_batches is not None and n_batches >= cfg.eval_max_batches:
                break
    _set_train_mode(sae, True)
    if n_batches == 0:
        return None
    return {
        "eval_loss": loss_sum / n_batches,
        "eval_fvu": fvu_sum / n_batches,
        "eval_auxk_loss": auxk_sum / n_batches,
        "eval_multi_topk_fvu": multi_topk_sum / n_batches,
        "eval_l0": l0_sum / n_batches,
        "eval_batches": float(n_batches),
    }


def _save_checkpoint(sae: JumpReluResidualSae, path: Path, stats: ResidualStats | None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    sae.save_to_disk(path)
    if stats is not None:
        save_residual_stats(path, stats)


def _grad_is_finite(sae: JumpReluResidualSae) -> bool:
    return all(param.grad is None or torch.isfinite(param.grad).all() for param in sae.parameters())


def _params_are_finite(sae: JumpReluResidualSae) -> bool:
    return all(torch.isfinite(param).all() for param in sae.parameters())


def train_residual_sae(cfg: ResidualSaeTrainConfig) -> dict[str, str | int | float]:
    device = resolve_visible_device(cfg.device)
    cache_dir = Path(cfg.residual_cache_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = _load_residual_shard_paths(cache_dir)
    assert cfg.eval_every_steps > 0, "eval_every_steps must be positive"
    train_shard_paths, val_shard_paths = _split_train_val_shards(
        shard_paths,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    first_residual = load_file(str(shard_paths[0]))["residual"]
    d_in = int(first_residual.shape[1])

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    logger.info(
        "Starting Qwen3.5-9B residual SAE training: cache_dir=%s, output_dir=%s, d_in=%d, n_shards=%d, "
        "train_shards=%d, val_shards=%d, batch_size=%d, lr=%s, weight_decay=%s, lr_warmup_steps=%d, "
        "grad_clip_norm=%s, max_epochs=%d, max_steps=%s, architecture=%s, "
        "jump_expansion_factor=%s, jump_l0_coefficient=%s, auxk_alpha=%s, dead_feature_threshold=%d, "
        "multi_topk_loss_weight=%s, normalize_residuals=%s, device=%s",
        cache_dir,
        output_dir,
        d_in,
        len(shard_paths),
        len(train_shard_paths),
        len(val_shard_paths),
        cfg.batch_size,
        cfg.lr,
        cfg.weight_decay,
        cfg.lr_warmup_steps,
        cfg.grad_clip_norm,
        cfg.max_epochs,
        cfg.max_steps,
        cfg.sae_architecture,
        cfg.jump_relu.expansion_factor,
        cfg.jump_relu.l0_coefficient,
        cfg.auxk_alpha,
        cfg.dead_feature_threshold,
        cfg.multi_topk_loss_weight,
        cfg.normalize_residuals,
        device,
    )

    residual_stats = None
    if cfg.normalize_residuals:
        residual_stats = _compute_residual_stats(train_shard_paths, cfg.normalization_eps)
        logger.info(
            "Residual normalization enabled: mean_abs=%.4f std_mean=%.4f std_min=%.4f std_max=%.4f",
            float(residual_stats.mean.abs().mean().item()),
            float(residual_stats.std.mean().item()),
            float(residual_stats.std.min().item()),
            float(residual_stats.std.max().item()),
        )

    sae = JumpReluResidualSae(d_in=d_in, cfg=cfg.jump_relu, device=device, dtype=torch.float32)
    logger.info(
        "Initializing decoder bias from first %d tokens",
        cfg.init_geometric_median_tokens,
    )
    _initialize_decoder_bias(sae, train_shard_paths, cfg.init_geometric_median_tokens, residual_stats)
    logger.info("Decoder bias initialized. SAE has %d latents.", sae.num_latents)

    optimizer = torch.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.lr_warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / cfg.lr_warmup_steps),
        )
    else:
        scheduler = None
    metrics_path = output_dir / "train_metrics.jsonl"
    with (output_dir / "config.json").open("w") as fd:
        json.dump(cfg.to_dict(), fd, indent=2)
    if residual_stats is not None:
        save_residual_stats(output_dir, residual_stats)

    global_step = 0
    best_loss = math.inf
    best_eval_loss = math.inf
    best_step = 0
    best_eval_step = 0
    num_tokens_since_fired = torch.zeros(sae.num_latents, dtype=torch.long, device=device)
    _set_train_mode(sae, True)
    t_start = time.monotonic()
    t_last_log = t_start
    for epoch in range(cfg.max_epochs):
        logger.info("Starting epoch %d / %d", epoch + 1, cfg.max_epochs)
        epoch_loss_sum = 0.0
        epoch_fvu_sum = 0.0
        epoch_steps = 0
        for batch, shard_name in _iter_residual_batches(
            train_shard_paths,
            batch_size=cfg.batch_size,
            device=device,
            seed=cfg.seed,
            epoch=epoch,
        ):
            batch = _normalize_batch(batch, residual_stats)
            next_step = global_step + 1
            assert torch.isfinite(batch).all(), f"Non-finite normalized residual batch at step={next_step}, shard={shard_name}"
            batch_total_variance = _batch_total_variance(batch)
            if batch.shape[0] < 2 or not math.isfinite(batch_total_variance) or batch_total_variance <= 0.0:
                logger.warning(
                    "Skipping degenerate batch at step=%d shard=%s batch_size=%d total_variance=%s",
                    next_step,
                    shard_name,
                    batch.shape[0],
                    batch_total_variance,
                )
                continue
            optimizer.zero_grad(set_to_none=True)
            dead_mask = num_tokens_since_fired > cfg.dead_feature_threshold if cfg.auxk_alpha > 0 else None
            out = sae(batch, dead_mask=dead_mask)
            loss = _loss_from_output(out, batch, cfg)
            if not torch.isfinite(loss):
                _save_checkpoint(sae, output_dir / "last_finite", residual_stats)
                raise FloatingPointError(f"Non-finite loss at step={next_step}, shard={shard_name}")
            loss.backward()
            if sae.cfg.normalize_decoder:
                sae.remove_gradient_parallel_to_decoder_directions()
            if cfg.grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), float("inf"))
            grad_norm_val = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            if not math.isfinite(grad_norm_val) or not _grad_is_finite(sae):
                _save_checkpoint(sae, output_dir / "last_finite", residual_stats)
                raise FloatingPointError(f"Non-finite gradients at step={next_step}, shard={shard_name}")
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if sae.cfg.normalize_decoder:
                sae.set_decoder_norm_to_unit_norm()
            if not _params_are_finite(sae):
                raise FloatingPointError(f"Non-finite parameters after optimizer step at step={next_step}, shard={shard_name}")

            global_step = next_step
            epoch_steps += 1
            loss_val = float(loss.item())
            fvu_val = float(out.fvu.item())
            epoch_loss_sum += loss_val
            epoch_fvu_sum += fvu_val
            param_stats = _parameter_stats(sae)
            if loss_val < best_loss:
                best_loss = loss_val
                best_step = global_step
                if not val_shard_paths and (global_step == 1 or global_step % cfg.save_every_steps == 0):
                    _save_checkpoint(sae, output_dir / "best_finite", residual_stats)

            did_fire = torch.zeros_like(num_tokens_since_fired, dtype=torch.bool)
            _mark_fired(did_fire, out)
            num_tokens_since_fired += batch.shape[0]
            num_tokens_since_fired[did_fire] = 0
            dead_pct = float((num_tokens_since_fired > cfg.dead_feature_threshold).float().mean().item())
            l0_val = _latent_l0(out)
            l0_loss_val = float(_l0_loss(out, batch).item())

            metrics = {
                "epoch": epoch,
                "step": global_step,
                "loss": loss_val,
                "fvu": fvu_val,
                "auxk_loss": float(out.auxk_loss.item()),
                "multi_topk_fvu": float(out.multi_topk_fvu.item()),
                "l0": l0_val,
                "l0_loss": l0_loss_val,
                "dead_pct": dead_pct,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "shard_name": shard_name,
                "grad_norm": grad_norm_val,
                **param_stats,
            }
            if val_shard_paths and (global_step == 1 or global_step % cfg.eval_every_steps == 0):
                eval_metrics = _evaluate_sae(
                    sae,
                    val_shard_paths,
                    batch_size=cfg.batch_size,
                    device=device,
                    stats=residual_stats,
                    cfg=cfg,
                )
                if eval_metrics is not None:
                    metrics.update(eval_metrics)
                    if eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        best_eval_step = global_step
                        _save_checkpoint(sae, output_dir / "best_finite", residual_stats)
                        logger.info(
                            "New best eval checkpoint: step=%d eval_loss=%.4f eval_fvu=%.4f eval_l0=%.2f",
                            global_step,
                            eval_metrics["eval_loss"],
                            eval_metrics["eval_fvu"],
                            eval_metrics["eval_l0"],
                        )
            with metrics_path.open("a") as fd:
                fd.write(json.dumps(metrics) + "\n")

            now = time.monotonic()
            if global_step % cfg.log_every_steps == 0 or (now - t_last_log) > 60:
                elapsed = now - t_start
                logger.info(
                    "step=%d epoch=%d loss=%.4f fvu=%.4f auxk=%.4f multi_topk_fvu=%.4f l0=%.2f dead_pct=%.4f "
                    "grad_norm=%.4f lr=%.2e b_dec_norm=%.4f W_dec_norm=%.4f shard=%s elapsed=%.0fs",
                    global_step,
                    epoch,
                    loss_val,
                    fvu_val,
                    float(out.auxk_loss.item()),
                    float(out.multi_topk_fvu.item()),
                    l0_val,
                    dead_pct,
                    grad_norm_val,
                    float(optimizer.param_groups[0]["lr"]),
                    param_stats["b_dec_norm"],
                    param_stats.get("W_dec_norm", float("nan")),
                    shard_name,
                    elapsed,
                )
                t_last_log = now

            if global_step % cfg.save_every_steps == 0:
                ckpt_path = output_dir / f"step_{global_step:06d}"
                _save_checkpoint(sae, ckpt_path, residual_stats)
                _save_checkpoint(sae, output_dir / "last_finite", residual_stats)
                if (not val_shard_paths and global_step == best_step) or (val_shard_paths and global_step == best_eval_step):
                    _save_checkpoint(sae, output_dir / "best_finite", residual_stats)
                logger.info("Saved checkpoint to %s", ckpt_path)

            if cfg.max_steps is not None and global_step >= cfg.max_steps:
                _save_checkpoint(sae, output_dir / "final", residual_stats)
                _save_checkpoint(sae, output_dir / "last_finite", residual_stats)
                if (not val_shard_paths and global_step == best_step) or (val_shard_paths and global_step == best_eval_step):
                    _save_checkpoint(sae, output_dir / "best_finite", residual_stats)
                logger.info(
                    "Reached max_steps=%d. Saved final model. best_step=%d best_loss=%.6f best_eval_step=%d "
                    "best_eval_loss=%.6f total_elapsed=%.0fs",
                    cfg.max_steps,
                    best_step,
                    best_loss,
                    best_eval_step,
                    best_eval_loss,
                    time.monotonic() - t_start,
                )
                return {"output_dir": str(output_dir), "steps": global_step, "d_in": d_in}

        logger.info(
            "Epoch %d complete: steps=%d avg_loss=%.4f avg_fvu=%.4f",
            epoch,
            epoch_steps,
            epoch_loss_sum / max(epoch_steps, 1),
            epoch_fvu_sum / max(epoch_steps, 1),
        )

    _save_checkpoint(sae, output_dir / "final", residual_stats)
    _save_checkpoint(sae, output_dir / "last_finite", residual_stats)
    if (not val_shard_paths and global_step == best_step) or (val_shard_paths and global_step == best_eval_step):
        _save_checkpoint(sae, output_dir / "best_finite", residual_stats)
    logger.info(
        "Training complete: steps=%d best_step=%d best_loss=%.6f best_eval_step=%d best_eval_loss=%.6f "
        "total_elapsed=%.0fs output_dir=%s",
        global_step,
        best_step,
        best_loss,
        best_eval_step,
        best_eval_loss,
        time.monotonic() - t_start,
        output_dir,
    )
    return {"output_dir": str(output_dir), "steps": global_step, "d_in": d_in}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ResidualSaeTrainConfig)
    result = train_residual_sae(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()