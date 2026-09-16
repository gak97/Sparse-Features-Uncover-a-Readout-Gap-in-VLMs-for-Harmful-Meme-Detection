
import json
import logging
import random
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file
from simple_parsing import ArgumentGenerationMode, parse

from gemma3_experiments.residual_sae.devices import resolve_visible_device
from gemma3_experiments.residual_sae.matryoshka import MatryoshkaJumpReluConfig, MatryoshkaJumpReluSae
from gemma3_experiments.residual_sae.normalization import ResidualStats, save_residual_stats
from gemma3_experiments.vendor_sae.utils import geometric_median


logger = logging.getLogger(__name__)


@dataclass
class MatryoshkaTrainConfig:
    residual_cache_dir: str = "./residual_sae_cache/residual"
    output_dir: str = "./residual_sae_ckpts_matryoshka"
    device: str = "cuda:0"

    expansion_factor: int = 8
    num_latents: int = 0
    prefix_sizes: list[int] = field(default_factory=lambda: [256, 1024, 4096, 16384])
    prefix_loss_weights: list[float] = field(default_factory=list)
    threshold_init: float = 0.1
    threshold_bandwidth: float = 0.1
    l0_coefficient: float = 3e-4
    normalize_decoder: bool = True

    batch_size: int = 4096
    lr: float = 5e-5
    weight_decay: float = 0.0
    lr_warmup_steps: int = 100
    grad_clip_norm: float | None = 1.0
    max_epochs: int = 1
    max_steps: int | None = None
    seed: int = 42

    val_fraction: float = 0.05
    eval_every_steps: int = 250
    eval_max_batches: int | None = 64
    save_every_steps: int = 500
    log_every_steps: int = 50
    init_geometric_median_tokens: int = 8192
    normalize_residuals: bool = True
    normalization_eps: float = 1e-5


def _read_shards(cache_dir: Path) -> list[Path]:
    shards = sorted(cache_dir.glob("residual_*.safetensors"))
    assert shards, f"No residual shards found in {cache_dir}"
    return shards


def _split_shards(shards: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    assert 0.0 <= val_fraction < 1.0
    if val_fraction == 0.0 or len(shards) < 2:
        return shards, []
    rng = random.Random(seed)
    shuffled = list(shards)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shards) * val_fraction))
    n_val = min(n_val, len(shards) - 1)
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def _iter_batches(shards: list[Path], *, batch_size: int, device: str, seed: int, epoch: int):
    rng = random.Random(seed + epoch)
    shuffled = list(shards)
    rng.shuffle(shuffled)
    for shard in shuffled:
        residual = load_file(str(shard))["residual"]
        shard_seed = zlib.crc32(shard.name.encode("utf-8"))
        gen = torch.Generator().manual_seed(seed + epoch * 1_000_003 + shard_seed)
        residual = residual[torch.randperm(residual.shape[0], generator=gen)]
        for start in range(0, residual.shape[0], batch_size):
            batch = residual[start : start + batch_size]
            if batch.numel():
                yield batch.to(device=device, dtype=torch.float32), shard.name


def _compute_stats(shards: list[Path], eps: float) -> ResidualStats:
    d_in = int(load_file(str(shards[0]))["residual"].shape[1])
    total = 0
    sum_x = torch.zeros(d_in, dtype=torch.float64)
    sum_x2 = torch.zeros(d_in, dtype=torch.float64)
    for shard in shards:
        x = load_file(str(shard))["residual"].to(torch.float64)
        total += int(x.shape[0])
        sum_x += x.sum(0)
        sum_x2 += x.square().sum(0)
    mean = sum_x / total
    var = (sum_x2 / total) - mean.square()
    std = torch.sqrt(torch.clamp(var, min=eps**2))
    return ResidualStats(mean=mean.float(), std=std.float(), eps=eps)


def _normalize(x: torch.Tensor, stats: ResidualStats | None) -> torch.Tensor:
    return stats.normalize(x) if stats is not None else x


def _init_decoder_bias(
    sae: MatryoshkaJumpReluSae,
    shards: list[Path],
    n_tokens: int,
    stats: ResidualStats | None,
) -> None:
    if n_tokens <= 0:
        return
    points: list[torch.Tensor] = []
    remaining = n_tokens
    for shard in shards:
        x = load_file(str(shard))["residual"]
        take = min(remaining, x.shape[0])
        if take > 0:
            points.append(x[:take])
            remaining -= take
        if remaining == 0:
            break
    if not points:
        return
    x = torch.cat(points, dim=0).to(device=sae.device, dtype=sae.dtype)
    x = _normalize(x, stats)
    sae.b_dec.data.copy_(geometric_median(x))


def _batch_variance_ok(x: torch.Tensor) -> bool:
    total_variance = (x - x.mean(0)).pow(2).sum()
    return bool(torch.isfinite(total_variance) and total_variance > 0)


@torch.no_grad()
def _evaluate(
    sae: MatryoshkaJumpReluSae,
    shards: list[Path],
    *,
    batch_size: int,
    device: str,
    stats: ResidualStats | None,
    max_batches: int | None,
) -> dict[str, float] | None:
    if not shards:
        return None
    sae.eval()
    sums: dict[str, float] = {"loss": 0.0, "fvu": 0.0, "l0": 0.0, "l0_loss": 0.0}
    prefix_sums = {k: 0.0 for k in sae.prefix_sizes}
    n = 0
    for batch, _ in _iter_batches(shards, batch_size=batch_size, device=device, seed=0, epoch=0):
        batch = _normalize(batch, stats)
        if not _batch_variance_ok(batch):
            continue
        out = sae(batch)
        sums["loss"] += float(out.loss.item())
        sums["fvu"] += float(out.fvu.item())
        sums["l0"] += float(out.l0.item())
        sums["l0_loss"] += float(out.l0_loss.item())
        for k, v in out.prefix_fvu.items():
            prefix_sums[k] += float(v.item())
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    sae.train()
    if n == 0:
        return None
    metrics = {f"eval_{k}": v / n for k, v in sums.items()}
    metrics.update({f"eval_prefix_fvu_{k}": v / n for k, v in prefix_sums.items()})
    return metrics


def _save_checkpoint(sae: MatryoshkaJumpReluSae, path: Path, stats: ResidualStats | None) -> None:
    sae.save_to_disk(path)
    if stats is not None:
        save_residual_stats(path, stats)


def train_matryoshka_sae(cfg: MatryoshkaTrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cache_dir = Path(cfg.residual_cache_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as fd:
        json.dump(asdict(cfg), fd, indent=2)

    device = resolve_visible_device(cfg.device)
    shards = _read_shards(cache_dir)
    train_shards, val_shards = _split_shards(shards, cfg.val_fraction, cfg.seed)
    d_in = int(load_file(str(shards[0]))["residual"].shape[1])
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    stats = _compute_stats(train_shards, cfg.normalization_eps) if cfg.normalize_residuals else None
    if stats is not None:
        save_residual_stats(output_dir, stats)

    msae_cfg = MatryoshkaJumpReluConfig(
        expansion_factor=cfg.expansion_factor,
        num_latents=cfg.num_latents,
        prefix_sizes=cfg.prefix_sizes,
        prefix_loss_weights=cfg.prefix_loss_weights,
        threshold_init=cfg.threshold_init,
        threshold_bandwidth=cfg.threshold_bandwidth,
        l0_coefficient=cfg.l0_coefficient,
        normalize_decoder=cfg.normalize_decoder,
    )
    sae = MatryoshkaJumpReluSae(d_in=d_in, cfg=msae_cfg, device=device, dtype=torch.float32)
    _init_decoder_bias(sae, train_shards, cfg.init_geometric_median_tokens, stats)

    logger.info(
        "Training MSAE: d_in=%d num_latents=%d prefixes=%s weights=%s train_shards=%d val_shards=%d",
        d_in, sae.num_latents, sae.prefix_sizes, sae.prefix_loss_weights, len(train_shards), len(val_shards),
    )
    optimizer = torch.optim.AdamW(sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.lr_warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / cfg.lr_warmup_steps)
        )

    metrics_path = output_dir / "train_metrics.jsonl"
    best_eval_loss = float("inf")
    step = 0
    start_time = time.time()
    stop = False

    for epoch in range(cfg.max_epochs):
        for batch, shard_name in _iter_batches(
            train_shards, batch_size=cfg.batch_size, device=device, seed=cfg.seed, epoch=epoch
        ):
            batch = _normalize(batch, stats)
            if not _batch_variance_ok(batch):
                continue
            optimizer.zero_grad(set_to_none=True)
            out = sae(batch)
            if not torch.isfinite(out.loss):
                raise RuntimeError(f"Non-finite MSAE loss at step={step} shard={shard_name}")
            out.loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip_norm)
            sae.remove_gradient_parallel_to_decoder_directions()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if cfg.normalize_decoder:
                sae.set_decoder_norm_to_unit_norm()

            step += 1
            if step % cfg.log_every_steps == 0:
                row = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(out.loss.item()),
                    "fvu": float(out.fvu.item()),
                    "l0": float(out.l0.item()),
                    "l0_loss": float(out.l0_loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_sec": round(time.time() - start_time, 2),
                    **{f"prefix_fvu_{k}": float(v.item()) for k, v in out.prefix_fvu.items()},
                }
                with metrics_path.open("a") as fd:
                    fd.write(json.dumps(row) + "\n")
                logger.info(
                    "step=%d loss=%.4f fvu=%.4f l0=%.1f",
                    step, row["loss"], row["fvu"], row["l0"],
                )

            if cfg.eval_every_steps and step % cfg.eval_every_steps == 0:
                eval_metrics = _evaluate(
                    sae, val_shards, batch_size=cfg.batch_size, device=device, stats=stats,
                    max_batches=cfg.eval_max_batches,
                )
                if eval_metrics:
                    row = {"step": step, "epoch": epoch, **eval_metrics}
                    with metrics_path.open("a") as fd:
                        fd.write(json.dumps(row) + "\n")
                    if eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        _save_checkpoint(sae, output_dir / "best_eval", stats)
                        logger.info("New best_eval at step=%d eval_loss=%.4f", step, best_eval_loss)

            if cfg.save_every_steps and step % cfg.save_every_steps == 0:
                _save_checkpoint(sae, output_dir / f"step_{step:06d}", stats)

            if cfg.max_steps is not None and step >= cfg.max_steps:
                stop = True
                break
        if stop:
            break

    _save_checkpoint(sae, output_dir / "final", stats)
    logger.info("Finished MSAE training at step=%d; final saved to %s", step, output_dir / "final")


def main() -> None:
    cfg = parse(MatryoshkaTrainConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    train_matryoshka_sae(cfg)


if __name__ == "__main__":
    main()
