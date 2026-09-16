

import concurrent.futures
import fcntl
import json
import logging
import multiprocessing as mp
import re
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from simple_parsing import parse

from gemma3_experiments.residual_sae.datasets import MemeSample, load_meme_samples

from ..config import DenseActivationCacheConfig
from ..qwen import (
    capture_task_hidden_states,
    load_qwen3_model,
    selection_mask,
    token_roles,
)

logger = logging.getLogger(__name__)

_DENSE_SHARD_RE = re.compile(
    r"^dense(?:_(?P<worker>w\d+))?_(?P<index>\d{5})\.safetensors$"
)


def _prepare_output_dir(path: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("overwrite and resume cannot both be enabled.")
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _save_jsonl(path: Path, rows: list[dict[str, Any]], *, lock: bool = False) -> None:
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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as fd:
        json.dump(payload, fd, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as fd:
        return json.load(fd)


def _parse_shard_name(shard_name: str) -> tuple[str | None, int] | None:
    match = _DENSE_SHARD_RE.match(Path(shard_name).name)
    if match is None:
        return None
    return match.group("worker"), int(match.group("index"))


def _make_shard_name(shard_index: int, worker_tag: str | None = None) -> str:
    wp = "" if worker_tag is None else f"_{worker_tag}"
    return f"dense{wp}_{shard_index:05d}.safetensors"


def _next_shard_index(output_dir: Path, *, worker_tag: str | None = None) -> int:
    next_index = 0
    for fpath in output_dir.glob("dense_*.safetensors"):
        parsed = _parse_shard_name(fpath.name)
        if parsed is None:
            continue
        swt, sidx = parsed
        if swt != worker_tag:
            continue
        next_index = max(next_index, sidx + 1)
    return next_index


def _get_parallel_devices(requested_device: str) -> list[str]:
    if not requested_device.startswith("cuda") or not torch.cuda.is_available():
        return [requested_device]
    n = torch.cuda.device_count()
    if n <= 1:
        return [requested_device]
    return [f"cuda:{i}" for i in range(n)]


def _partition_round_robin(
    items: list[Any], n_parts: int
) -> list[list[Any]]:
    parts: list[list[Any]] = [[] for _ in range(n_parts)]
    for i, item in enumerate(items):
        parts[i % n_parts].append(item)
    return [p for p in parts if p]


def _configure_worker_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _flush_dense_shard(
    output_dir: Path,
    manifest_path: Path,
    shard_index: int,
    entries: list[dict[str, Any]],
    hidden_buf: list[torch.Tensor],
    token_id_buf: list[torch.Tensor],
    role_buf: list[torch.Tensor],
    *,
    worker_tag: str | None = None,
    lock_manifest: bool = False,
) -> None:
    assert entries, "Cannot flush an empty shard."
    shard_name = _make_shard_name(shard_index, worker_tag)
    n_tokens = sum(int(t.shape[0]) for t in hidden_buf)
    save_file(
        {
            "hidden": torch.cat(hidden_buf, dim=0),
            "token_ids": torch.cat(token_id_buf, dim=0),
            "token_roles": torch.cat(role_buf, dim=0),
        },
        str(output_dir / shard_name),
    )
    for entry in entries:
        entry["shard_file"] = shard_name
    _save_jsonl(manifest_path, entries, lock=lock_manifest)
    logger.info(
        "Wrote dense shard %s with %d samples/%d tokens to %s",
        shard_name,
        len(entries),
        n_tokens,
        output_dir,
    )


def _run_dense_worker(
    cfg: DenseActivationCacheConfig,
    samples: list[MemeSample],
    output_dir_str: str,
    manifest_path_str: str,
    *,
    worker_tag: str | None,
    model_device: str,
    lock_manifest: bool,
) -> dict[str, int]:
    _configure_worker_logging()
    output_dir = Path(output_dir_str)
    manifest_path = Path(manifest_path_str)
    worker_cfg = replace(cfg, qwen=replace(cfg.qwen, model_device=model_device))
    shard_index = _next_shard_index(output_dir, worker_tag=worker_tag)
    model, processor = load_qwen3_model(worker_cfg.qwen)
    shard_entries: list[dict[str, Any]] = []
    hidden_buf: list[torch.Tensor] = []
    token_id_buf: list[torch.Tensor] = []
    role_buf: list[torch.Tensor] = []
    n_written_shards = 0
    total_selected_tokens = 0
    progress_every = max(1, len(samples) // 20) if samples else 1
    current_group: tuple[str, str, str] | None = None
    worker_name = worker_tag or "serial"

    logger.info(
        "Worker %s on %s: %d samples, next shard index=%d",
        worker_name,
        model_device,
        len(samples),
        shard_index,
    )

    for sample_i, sample in enumerate(samples, start=1):
        sg = (sample.dataset_name, sample.split, sample.task_name)
        if sg != current_group:
            current_group = sg
            logger.info(
                "Worker %s caching %s/%s/%s (%d/%d) id=%s",
                worker_name,
                sample.dataset_name,
                sample.split,
                sample.task_name,
                sample_i,
                len(samples),
                sample.sample_id,
            )

        run = capture_task_hidden_states(model, processor, sample, worker_cfg.qwen)
        roles = token_roles(
            run.hidden.shape[0], run.image_span, run.input_len
        )
        mask = selection_mask(roles, worker_cfg.qwen.token_selection)
        selected_hidden = run.hidden[mask].contiguous()
        selected_token_ids = run.output_ids[mask].to(torch.int64)
        selected_roles = roles[mask]
        total_selected_tokens += int(selected_hidden.shape[0])
        offset = sum(t.shape[0] for t in hidden_buf)
        shard_entries.append(
            {
                **sample.to_manifest_dict(),
                "prompt_text": run.prompt_text,
                "generated_text": run.generated_text,
                "input_len": run.input_len,
                "image_span": list(run.image_span),
                "offset": offset,
                "length": int(selected_hidden.shape[0]),
                "selected_token_selection": worker_cfg.qwen.token_selection,
            }
        )
        hidden_buf.append(selected_hidden)
        token_id_buf.append(selected_token_ids)
        role_buf.append(selected_roles)

        if sample_i % progress_every == 0 or sample_i == len(samples):
            logger.info(
                "Worker %s progress: %d/%d samples, %d tokens buffered",
                worker_name,
                sample_i,
                len(samples),
                sum(int(t.shape[0]) for t in hidden_buf),
            )

        if len(shard_entries) >= cfg.shard_size:
            _flush_dense_shard(
                output_dir,
                manifest_path,
                shard_index,
                shard_entries,
                hidden_buf,
                token_id_buf,
                role_buf,
                worker_tag=worker_tag,
                lock_manifest=lock_manifest,
            )
            shard_index += 1
            n_written_shards += 1
            shard_entries = []
            hidden_buf = []
            token_id_buf = []
            role_buf = []

    if shard_entries:
        _flush_dense_shard(
            output_dir,
            manifest_path,
            shard_index,
            shard_entries,
            hidden_buf,
            token_id_buf,
            role_buf,
            worker_tag=worker_tag,
            lock_manifest=lock_manifest,
        )
        n_written_shards += 1

    return {
        "n_samples": len(samples),
        "n_written_shards": n_written_shards,
        "selected_tokens": total_selected_tokens,
    }


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_dense_activation_cache(
    cfg: DenseActivationCacheConfig,
    samples: list[MemeSample] | None = None,
) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    _prepare_output_dir(output_dir, cfg.overwrite, cfg.resume)

    logger.info(
        "Dense cache: output_dir=%s shard_size=%d overwrite=%s resume=%s "
        "model=%s hook=%s token_selection=%s capture_generation=%s",
        output_dir,
        cfg.shard_size,
        cfg.overwrite,
        cfg.resume,
        cfg.qwen.model_id,
        cfg.qwen.hook_module_path,
        cfg.qwen.token_selection,
        cfg.qwen.capture_generation,
    )

    dataset_cfg = cfg.dataset
    if samples is None:
        if cfg.skip_mmhs150k:
            datasets = [d for d in cfg.dataset.datasets if d != "MMHS150K"]
            dataset_cfg = replace(cfg.dataset, datasets=datasets)
            logger.info("MMHS150K skip enabled")
        samples = load_meme_samples(dataset_cfg)
    else:
        logger.info("Using %d caller-provided samples for dense caching", len(samples))
    logger.info("Loaded %d samples for dense caching", len(samples))

    config_payload = {
        "dataset": dataset_cfg.to_dict(),
        "qwen": cfg.qwen.to_dict(),
        "n_samples": len(samples),
        "skip_mmhs150k": cfg.skip_mmhs150k,
    }
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "manifest.jsonl"
    existing_rows: list[dict[str, Any]] = []
    n_resumed_samples = 0

    if cfg.resume and (
        config_path.exists()
        or manifest_path.exists()
        or any(output_dir.glob("dense_*.safetensors"))
    ):
        existing_config = _load_json(config_path)
        if (
            existing_config.get("qwen", {}).get("model_id")
            != cfg.qwen.model_id
        ):
            raise ValueError("Resume conflict: model_id mismatch")
        if manifest_path.exists():
            existing_rows = _load_manifest(manifest_path)
        cached_keys = {
            (r["dataset_name"], r["split"], r["task_name"], r["sample_id"])
            for r in existing_rows
        }
        n_resumed_samples = len(cached_keys)
        samples = [
            s
            for s in samples
            if (
                s.dataset_name,
                s.split,
                s.task_name,
                s.sample_id,
            )
            not in cached_keys
        ]
        logger.info(
            "Resuming: %d already cached, %d remaining",
            n_resumed_samples,
            len(samples),
        )
    else:
        _save_json(config_path, config_payload)

    if not samples:
        logger.info("Dense cache already complete")
        return config_payload

    devices = _get_parallel_devices(cfg.qwen.model_device)
    n_workers = min(len(devices), len(samples))
    logger.info(
        "Using %d worker(s) on %s", n_workers, ",".join(devices[:n_workers])
    )

    if n_workers == 1:
        result = _run_dense_worker(
            cfg,
            samples,
            str(output_dir),
            str(manifest_path),
            worker_tag=None,
            model_device=devices[0],
            lock_manifest=False,
        )
        n_written_shards = result["n_written_shards"]
        total_selected_tokens = result["selected_tokens"]
    else:
        partitions = _partition_round_robin(samples, n_workers)
        worker_devices = devices[: len(partitions)]
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(partitions), mp_context=ctx
        ) as executor:
            futures = [
                executor.submit(
                    _run_dense_worker,
                    cfg,
                    ps,
                    str(output_dir),
                    str(manifest_path),
                    worker_tag=f"w{i:02d}",
                    model_device=wd,
                    lock_manifest=True,
                )
                for i, (ps, wd) in enumerate(zip(partitions, worker_devices))
            ]
            results = [f.result() for f in futures]
        n_written_shards = sum(r["n_written_shards"] for r in results)
        total_selected_tokens = sum(r["selected_tokens"] for r in results)

    logger.info(
        "Dense cache complete: %d new samples, %d resumed, %d tokens, %d shards",
        len(samples),
        n_resumed_samples,
        total_selected_tokens,
        n_written_shards,
    )
    return config_payload


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse(DenseActivationCacheConfig)
    run_dense_activation_cache(cfg)


if __name__ == "__main__":
    main()