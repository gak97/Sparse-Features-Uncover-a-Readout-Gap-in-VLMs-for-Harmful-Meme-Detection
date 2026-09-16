import concurrent.futures
import fcntl
import json
import logging
import multiprocessing as mp
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .config import DenseActivationCacheConfig, ResidualCacheConfig
from .datasets import MemeSample, load_meme_samples
from .devices import resolve_visible_device
from .gemma import GemmaSae, capture_task_hidden_states, load_gemma3_model


logger = logging.getLogger(__name__)

ROLE_PROMPT = 0
ROLE_IMAGE = 1
ROLE_GENERATED = 2
_DENSE_SHARD_RE = re.compile(r"^dense(?:_(?P<worker>w\d+))?_(?P<index>\d{5})\.safetensors$")
_RESIDUAL_SHARD_RE = re.compile(r"^residual(?:_(?P<worker>w\d+))?_(?P<index>\d{5})\.safetensors$")


def _prepare_output_dir(path: Path, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("overwrite and resume cannot both be enabled.")
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _token_roles(n_tokens: int, image_span: tuple[int, int], input_len: int) -> torch.Tensor:
    roles = torch.full((n_tokens,), ROLE_PROMPT, dtype=torch.int64)
    start, end = image_span
    roles[start:end] = ROLE_IMAGE
    if input_len < n_tokens:
        roles[input_len:] = ROLE_GENERATED
    return roles


def _selection_mask(roles: torch.Tensor, selection: str) -> torch.Tensor:
    if selection == "all":
        return torch.ones_like(roles, dtype=torch.bool)
    if selection == "image":
        return roles == ROLE_IMAGE
    if selection == "generated":
        return roles == ROLE_GENERATED
    if selection == "image_and_generated":
        return (roles == ROLE_IMAGE) | (roles == ROLE_GENERATED)
    raise ValueError(f"Unsupported token selection: {selection}")


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
        payload = json.load(fd)
    assert isinstance(payload, dict), f"Expected a JSON object at {path}"
    return payload


def _parse_shard_name(shard_name: str, *, is_dense: bool) -> tuple[str | None, int] | None:
    match = (_DENSE_SHARD_RE if is_dense else _RESIDUAL_SHARD_RE).match(Path(shard_name).name)
    if match is None:
        return None
    return match.group("worker"), int(match.group("index"))


def _make_shard_name(kind: str, shard_index: int, worker_tag: str | None = None) -> str:
    worker_prefix = "" if worker_tag is None else f"_{worker_tag}"
    return f"{kind}{worker_prefix}_{shard_index:05d}.safetensors"


def _make_residual_shard_name(dense_shard_name: str) -> str:
    dense_fname = Path(dense_shard_name).name
    assert dense_fname.startswith("dense_"), f"Unexpected dense shard name: {dense_shard_name}"
    return f"residual_{dense_fname.removeprefix('dense_')}"


def _next_shard_index(output_dir: Path, *, is_dense: bool, worker_tag: str | None = None) -> int:
    prefix = "dense" if is_dense else "residual"
    next_index = 0
    for shard_fpath in output_dir.glob(f"{prefix}_*.safetensors"):
        parsed = _parse_shard_name(shard_fpath.name, is_dense=is_dense)
        if parsed is None:
            continue
        shard_worker_tag, shard_index = parsed
        if shard_worker_tag != worker_tag:
            continue
        next_index = max(next_index, shard_index + 1)
    return next_index


def _warn_about_orphan_shards(output_dir: Path, manifest_rows: list[dict[str, Any]], *, is_dense: bool) -> None:
    prefix = "dense" if is_dense else "residual"
    manifest_shards = {row["shard_file"] for row in manifest_rows}
    orphan_shards = sorted(
        shard_fpath.name
        for shard_fpath in output_dir.glob(f"{prefix}_*.safetensors")
        if shard_fpath.name not in manifest_shards
    )
    if orphan_shards:
        logger.warning(
            "Found %d %s shard files with no manifest rows. They will be ignored for resume: %s",
            len(orphan_shards),
            prefix,
            ", ".join(orphan_shards[:5]),
        )


_RESUME_IDENTITY_KEYS = ("dataset", "gemma", "dense_cache_dir", "n_samples")


def _validate_resume_config(path: Path, expected: dict[str, Any], *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Resume requested for {label}, but config file is missing: {path}")
    existing = _load_json(path)
    keys_to_check = [k for k in _RESUME_IDENTITY_KEYS if k in expected]
    mismatches = {k: (existing.get(k), expected[k]) for k in keys_to_check if existing.get(k) != expected[k]}
    if mismatches:
        detail = "; ".join(f"{k}: existing={v[0]!r} vs current={v[1]!r}" for k, v in mismatches.items())
        raise ValueError(
            f"Resume requested for {label}, but existing config at {path} does not match the current arguments: {detail}"
        )


def _validate_manifest_shards(output_dir: Path, manifest_rows: list[dict[str, Any]], *, is_dense: bool) -> None:
    prefix = "dense" if is_dense else "residual"
    missing = sorted(
        {
            row["shard_file"]
            for row in manifest_rows
            if _parse_shard_name(row["shard_file"], is_dense=is_dense) is not None
            and not (output_dir / row["shard_file"]).exists()
        }
    )
    if missing:
        raise FileNotFoundError(
            f"Resume requested for {prefix} cache, but manifest references missing shard files: {', '.join(missing[:5])}"
        )
    _warn_about_orphan_shards(output_dir, manifest_rows, is_dense=is_dense)


def _sample_manifest_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        row["dataset_name"],
        row["split"],
        row["task_name"],
        row["sample_id"],
    )


def _get_parallel_devices(requested_device: str) -> list[str]:
    if not requested_device.startswith("cuda") or not torch.cuda.is_available():
        return [requested_device]
    n_devices = torch.cuda.device_count()
    if n_devices <= 1:
        return [resolve_visible_device(requested_device)]
    return [f"cuda:{i}" for i in range(n_devices)]


def _partition_round_robin(items: list[Any], n_parts: int) -> list[list[Any]]:
    partitions: list[list[Any]] = [[] for _ in range(n_parts)]
    for i, item in enumerate(items):
        partitions[i % n_parts].append(item)
    return partitions


def _configure_worker_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


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
    shard_name = _make_shard_name("dense", shard_index, worker_tag)
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
        "Wrote dense shard %s with %d samples and %d selected tokens to %s",
        shard_name,
        len(entries),
        n_tokens,
        output_dir,
    )


def _format_counter(counter: Counter[str]) -> str:
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def _get_dense_dataset_cfg(cfg: DenseActivationCacheConfig):
    if not cfg.skip_mmhs150k:
        return cfg.dataset

    datasets = [name for name in cfg.dataset.datasets if name != "MMHS150K"]
    logger.info("MMHS150K skip enabled for dense caching")
    return replace(cfg.dataset, datasets=datasets)


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
    worker_cfg = replace(cfg, gemma=replace(cfg.gemma, model_device=model_device))
    shard_index = _next_shard_index(output_dir, is_dense=True, worker_tag=worker_tag)
    model, processor = load_gemma3_model(worker_cfg.gemma)
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
        "Dense worker %s starting on %s with %d samples and next shard index=%d",
        worker_name,
        model_device,
        len(samples),
        shard_index,
    )
    for sample_i, sample in enumerate(samples, start=1):
        sample_group = (sample.dataset_name, sample.split, sample.task_name)
        if sample_group != current_group:
            current_group = sample_group
            logger.info(
                "Dense worker %s now caching dataset=%s split=%s task=%s (%d/%d), starting sample_id=%s",
                worker_name,
                sample.dataset_name,
                sample.split,
                sample.task_name,
                sample_i,
                len(samples),
                sample.sample_id,
            )
        run = capture_task_hidden_states(model, processor, sample, worker_cfg.gemma)
        roles = _token_roles(run.hidden.shape[0], run.image_span, run.input_len)
        mask = _selection_mask(roles, worker_cfg.gemma.token_selection)
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
                "selected_token_selection": worker_cfg.gemma.token_selection,
            }
        )
        hidden_buf.append(selected_hidden)
        token_id_buf.append(selected_token_ids)
        role_buf.append(selected_roles)
        if sample_i == 1 or sample_i == len(samples) or sample_i % progress_every == 0:
            logger.info(
                "Dense worker %s progress: samples=%d/%d, current_dataset=%s, split=%s, sample_id=%s, selected_tokens=%d, buffered_samples=%d, buffered_tokens=%d",
                worker_name,
                sample_i,
                len(samples),
                sample.dataset_name,
                sample.split,
                sample.sample_id,
                int(selected_hidden.shape[0]),
                len(shard_entries),
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


def run_dense_activation_cache(cfg: DenseActivationCacheConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    _prepare_output_dir(output_dir, cfg.overwrite, cfg.resume)
    dataset_cfg = _get_dense_dataset_cfg(cfg)
    logger.info(
        "Starting dense activation cache: output_dir=%s, shard_size=%d, overwrite=%s, resume=%s, model=%s, hook_module=%s, token_selection=%s, capture_generation=%s, skip_mmhs150k=%s",
        output_dir,
        cfg.shard_size,
        cfg.overwrite,
        cfg.resume,
        cfg.gemma.model_id,
        cfg.gemma.hook_module_path,
        cfg.gemma.token_selection,
        cfg.gemma.capture_generation,
        cfg.skip_mmhs150k,
    )
    logger.info(
        "Dense activation caching uses model_device workers derived from visible GPUs. sae_device=%s is unused in this stage.",
        cfg.gemma.sae_device,
    )
    logger.info(
        "Requested datasets: root=%s, datasets=%s, splits=%s, sample_limit=%s, include_unlabeled=%s",
        dataset_cfg.root,
        ",".join(dataset_cfg.datasets),
        ",".join(dataset_cfg.splits),
        dataset_cfg.sample_limit,
        dataset_cfg.include_unlabeled,
    )
    samples = load_meme_samples(dataset_cfg)
    dataset_counts = Counter(sample.dataset_name for sample in samples)
    split_counts = Counter(sample.split for sample in samples)
    task_counts = Counter(sample.task_name for sample in samples)
    logger.info("Loaded %d samples for dense caching", len(samples))
    if samples:
        logger.info("Dense cache dataset counts: %s", _format_counter(dataset_counts))
        logger.info("Dense cache split counts: %s", _format_counter(split_counts))
        logger.info("Dense cache task counts: %s", _format_counter(task_counts))
    else:
        logger.warning("No samples were loaded for dense caching")

    config_payload = {
        "dataset": dataset_cfg.to_dict(),
        "gemma": cfg.gemma.to_dict(),
        "n_samples": len(samples),
        "skip_mmhs150k": cfg.skip_mmhs150k,
    }
    config_path = output_dir / "config.json"
    manifest_path = output_dir / "manifest.jsonl"
    existing_rows: list[dict[str, Any]] = []
    n_resumed_samples = 0
    if cfg.resume and (config_path.exists() or manifest_path.exists() or any(output_dir.glob("dense_*.safetensors"))):
        _validate_resume_config(config_path, config_payload, label="dense activation cache")
        if manifest_path.exists():
            existing_rows = _load_manifest(manifest_path)
            _validate_manifest_shards(output_dir, existing_rows, is_dense=True)
        cached_sample_keys = {_sample_manifest_key(row) for row in existing_rows}
        n_resumed_samples = len(cached_sample_keys)
        samples = [
            sample
            for sample in samples
            if (sample.dataset_name, sample.split, sample.task_name, sample.sample_id) not in cached_sample_keys
        ]
        logger.info(
            "Resuming dense cache with %d completed samples and %d remaining samples",
            n_resumed_samples,
            len(samples),
        )
    else:
        _save_json(config_path, config_payload)

    if not samples:
        logger.info(
            "Dense activation cache already complete: resumed_samples=%d, output_dir=%s",
            n_resumed_samples,
            output_dir,
        )
        return config_payload

    devices = _get_parallel_devices(cfg.gemma.model_device)
    n_workers = min(len(devices), len(samples))
    logger.info("Dense activation cache will use %d worker(s) on devices=%s", n_workers, ",".join(devices[:n_workers]))

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
        partitions = [part for part in _partition_round_robin(samples, n_workers) if part]
        worker_devices = devices[: len(partitions)]
        ctx = mp.get_context("spawn")
        futures = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(partitions),
            mp_context=ctx,
        ) as executor:
            for worker_i, (worker_samples, worker_device) in enumerate(zip(partitions, worker_devices, strict=True)):
                futures.append(
                    executor.submit(
                        _run_dense_worker,
                        cfg,
                        worker_samples,
                        str(output_dir),
                        str(manifest_path),
                        worker_tag=f"w{worker_i:02d}",
                        model_device=worker_device,
                        lock_manifest=True,
                    )
                )
            results = [future.result() for future in futures]
        n_written_shards = sum(result["n_written_shards"] for result in results)
        total_selected_tokens = sum(result["selected_tokens"] for result in results)

    logger.info(
        "Dense activation cache complete: new_samples=%d, resumed_samples=%d, selected_tokens=%d, new_shards=%d, output_dir=%s",
        len(samples),
        n_resumed_samples,
        total_selected_tokens,
        n_written_shards,
        output_dir,
    )
    return config_payload


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
    worker_cfg = replace(cfg, gemma=replace(cfg.gemma, sae_device=sae_device))
    base_sae = GemmaSae.load_from_hub(
        worker_cfg.gemma.sae_repo_id,
        worker_cfg.gemma.sae_hookpoint,
        device=worker_cfg.gemma.sae_device,
    )
    n_written_shards = 0
    for shard_name, shard_rows in sorted(shard_rows_by_name.items()):
        dense_fpath = dense_dir / shard_name
        out_name = _make_residual_shard_name(shard_name)
        with safe_open(str(dense_fpath), framework="pt", device="cpu") as fd:
            hidden = fd.get_tensor("hidden")
            token_ids = fd.get_tensor("token_ids")
            token_roles = fd.get_tensor("token_roles")

        residual_chunks: list[torch.Tensor] = []
        recon_error_sum = 0.0
        for start in range(0, hidden.shape[0], worker_cfg.gemma.sae_chunk_size):
            chunk = hidden[start : start + worker_cfg.gemma.sae_chunk_size].to(worker_cfg.gemma.sae_device)
            _, recon = base_sae.reconstruct(chunk)
            residual = (chunk - recon).detach().cpu()
            recon_error_sum += float(residual.pow(2).sum().item())
            residual_chunks.append(residual)

        residual = torch.cat(residual_chunks, dim=0)
        save_file(
            {
                "residual": residual,
                "token_ids": token_ids,
                "token_roles": token_roles,
            },
            str(output_dir / out_name),
        )
        out_rows = []
        for row in shard_rows:
            updated = dict(row)
            updated["shard_file"] = out_name
            updated["base_sae_hookpoint"] = worker_cfg.gemma.sae_hookpoint
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


def compute_residual_shards(cfg: ResidualCacheConfig) -> dict[str, Any]:
    dense_dir = Path(cfg.dense_cache_dir)
    output_dir = Path(cfg.output_dir)
    _prepare_output_dir(output_dir, cfg.overwrite, cfg.resume)
    manifest = _load_manifest(dense_dir / "manifest.jsonl")
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_shard[row["shard_file"]].append(row)
    logger.info(
        "Residual caching uses SAE workers derived from visible GPUs. model_device=%s is unused in this stage.",
        cfg.gemma.model_device,
    )

    config_payload = {
        "dense_cache_dir": str(dense_dir),
        "gemma": cfg.gemma.to_dict(),
        "n_shards": len(by_shard),
    }
    config_path = output_dir / "config.json"
    output_manifest_path = output_dir / "manifest.jsonl"
    completed_shards: set[str] = set()
    if cfg.resume and (config_path.exists() or output_manifest_path.exists() or any(output_dir.glob("residual_*.safetensors"))):
        _validate_resume_config(config_path, config_payload, label="residual cache")
        if output_manifest_path.exists():
            output_rows = _load_manifest(output_manifest_path)
            _validate_manifest_shards(output_dir, output_rows, is_dense=False)
            completed_shards = {row["shard_file"] for row in output_rows}
        logger.info(
            "Resuming residual cache with %d completed shards and %d remaining shards",
            len(completed_shards),
            sum(1 for dense_shard_name in by_shard if _make_residual_shard_name(dense_shard_name) not in completed_shards),
        )
    else:
        _save_json(config_path, config_payload)

    remaining_shards = [
        shard_name for shard_name in sorted(by_shard) if _make_residual_shard_name(shard_name) not in completed_shards
    ]
    if not remaining_shards:
        logger.info("Residual cache already complete: output_dir=%s", output_dir)
        return {"output_dir": str(output_dir), "n_shards": len(by_shard), "n_written_shards": 0}

    devices = _get_parallel_devices(cfg.gemma.sae_device)
    n_workers = min(len(devices), len(remaining_shards))
    logger.info("Residual cache will use %d worker(s) on devices=%s", n_workers, ",".join(devices[:n_workers]))

    if n_workers == 1:
        result = _compute_residual_worker(
            cfg,
            str(dense_dir),
            str(output_dir),
            {shard_name: by_shard[shard_name] for shard_name in remaining_shards},
            sae_device=devices[0],
            lock_manifest=False,
        )
        n_written_shards = result["n_written_shards"]
    else:
        partitions = [part for part in _partition_round_robin(remaining_shards, n_workers) if part]
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
                        {shard_name: by_shard[shard_name] for shard_name in worker_shards},
                        sae_device=worker_device,
                        lock_manifest=True,
                    )
                )
            results = [future.result() for future in futures]
        n_written_shards = sum(result["n_written_shards"] for result in results)

    return {"output_dir": str(output_dir), "n_shards": len(by_shard), "n_written_shards": n_written_shards}
