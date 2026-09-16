

import concurrent.futures
import json
import logging
import multiprocessing as mp
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from simple_parsing import parse

from ..config import ResidualCacheConfig
from ..qwen import QwenSae

logger = logging.getLogger(__name__)


def _configure_worker_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _make_residual_shard_name(dense_shard_name: str) -> str:
    dense_fname = Path(dense_shard_name).name
    assert dense_fname.startswith("dense_"), f"Unexpected dense shard name: {dense_shard_name}"
    return f"residual_{dense_fname.removeprefix('dense_')}"


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compute_residual_worker(
    cfg: ResidualCacheConfig,
    dense_dir_str: str,
    output_dir_str: str,
    shard_rows_by_name: dict[str, list[dict[str, Any]]],
    *,
    sae_device: str,
    lock_manifest: bool,
) -> dict[str, int]:
    _configure_worker_logging()
    dense_dir = Path(dense_dir_str)
    output_dir = Path(output_dir_str)
    worker_cfg = replace(cfg, qwen=replace(cfg.qwen, sae_device=sae_device))
    base_sae = QwenSae.load_from_hub(
        worker_cfg.qwen.sae_repo_id,
        worker_cfg.qwen.sae_layer,
        device=worker_cfg.qwen.sae_device,
    )
    logger.info(
        "Residual worker using SAE from %s layer %d on %s",
        worker_cfg.qwen.sae_repo_id,
        worker_cfg.qwen.sae_layer,
        sae_device,
    )
    n_written_shards = 0
    for shard_name, shard_rows in sorted(shard_rows_by_name.items()):
        dense_fpath = dense_dir / shard_name
        out_name = _make_residual_shard_name(shard_name)
        with safe_open(str(dense_fpath), framework="pt", device="cpu") as fd:
            hidden = fd.get_tensor("hidden")

        residual_chunks: list[torch.Tensor] = []
        recon_error_sum = 0.0
        for start in range(0, hidden.shape[0], worker_cfg.qwen.sae_chunk_size):
            chunk = hidden[start : start + worker_cfg.qwen.sae_chunk_size].to(worker_cfg.qwen.sae_device)
            _, recon = base_sae.reconstruct(chunk)
            residual = (chunk - recon).detach().cpu()
            recon_error_sum += float(residual.pow(2).sum().item())
            residual_chunks.append(residual)

        residual = torch.cat(residual_chunks, dim=0)
        save_file(
            {"residual": residual},
            str(output_dir / out_name),
        )
        out_rows = []
        for row in shard_rows:
            updated = dict(row)
            updated["shard_file"] = out_name
            updated["base_sae_repo_id"] = worker_cfg.qwen.sae_repo_id
            updated["base_sae_layer"] = worker_cfg.qwen.sae_layer
            out_rows.append(updated)
        _save_jsonl(output_dir / "manifest.jsonl", out_rows, lock=lock_manifest)
        with (output_dir / f"{out_name}.meta.json").open("w") as fd:
            json.dump(
                {
                    "source_dense_shard": shard_name,
                    "output_shard": out_name,
                    "n_tokens": int(residual.shape[0]),
                    "d_model": int(residual.shape[1]),
                    "residual_l2_sum": recon_error_sum,
                },
                fd,
                indent=2,
            )
        n_written_shards += 1
    return {"n_written_shards": n_written_shards}


def _save_jsonl(path: Path, rows: list[dict[str, Any]], *, lock: bool = False) -> None:
    import fcntl
    with path.open("a") as fd:
        if lock:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        try:
            for row in rows:
                fd.write(json.dumps(row) + "\n")
            fd.flush()
        finally:
            if lock:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)


def _get_parallel_devices(requested_device: str) -> list[str]:
    if not requested_device.startswith("cuda") or not torch.cuda.is_available():
        return [requested_device]
    n_devices = torch.cuda.device_count()
    if n_devices <= 1:
        return [requested_device]
    return [f"cuda:{i}" for i in range(n_devices)]


def compute_residual_shards(cfg: ResidualCacheConfig) -> dict[str, Any]:
    dense_dir = Path(cfg.dense_cache_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(dense_dir / "manifest.jsonl")
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_shard[row["shard_file"]].append(row)

    logger.info(
        "Residual caching: dense_dir=%s, output_dir=%s, n_dense_shards=%d",
        dense_dir,
        output_dir,
        len(by_shard),
    )
    logger.info(
        "Residual caching uses SAE workers derived from visible GPUs. model_device=%s is unused in this stage.",
        cfg.qwen.model_device,
    )

    config_payload = {
        "dense_cache_dir": str(dense_dir),
        "qwen": cfg.qwen.to_dict(),
        "n_shards": len(by_shard),
    }

    devices = _get_parallel_devices(cfg.qwen.sae_device)
    n_workers = min(len(devices), len(by_shard))
    logger.info("Residual cache will use %d worker(s) on devices=%s", n_workers, ",".join(devices[:n_workers]))

    if n_workers == 1:
        result = _compute_residual_worker(
            cfg,
            str(dense_dir),
            str(output_dir),
            dict(by_shard),
            sae_device=devices[0],
            lock_manifest=False,
        )
        n_written_shards = result["n_written_shards"]
    else:
        remaining_shards = sorted(by_shard)
        partitions = [remaining_shards[i::n_workers] for i in range(n_workers)]
        partitions = [p for p in partitions if p]
        worker_devices = devices[: len(partitions)]
        ctx = mp.get_context("spawn")
        futures = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(partitions),
            mp_context=ctx,
        ) as executor:
            for worker_shards, worker_device in zip(partitions, worker_devices, strict=True):
                futures.append(
                    executor.submit(
                        _compute_residual_worker,
                        cfg,
                        str(dense_dir),
                        str(output_dir),
                        {s: by_shard[s] for s in worker_shards},
                        sae_device=worker_device,
                        lock_manifest=True,
                    )
                )
            results = [future.result() for future in futures]
        n_written_shards = sum(result["n_written_shards"] for result in results)

    logger.info(
        "Residual cache complete: output_dir=%s, n_shards=%d, n_written_shards=%d",
        output_dir,
        len(by_shard),
        n_written_shards,
    )
    return config_payload


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ResidualCacheConfig)
    compute_residual_shards(cfg)


if __name__ == "__main__":
    main()