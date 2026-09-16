

import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from simple_parsing import ArgumentGenerationMode, parse

logger = logging.getLogger(__name__)

# Task -> (task_key, relevant target-token words within DEFAULT_TARGET_TOKENS /
# target_token_ids.json). "yes"/"no" are included everywhere since they are the
# generic decision-routing tokens for all four constrained yes/no tasks.
TASK_TOKEN_MAP: dict[str, dict[str, Any]] = {
    "MAMI": {
        "task_key": "MAMI::misogyny_and_subtypes::label",
        "tokens": ["yes", "no", "misogynistic", "misogyny", "women"],
    },
    "HarMeme": {
        "task_key": "HarMeme::harmfulness_and_target::harmfulness",
        "tokens": ["yes", "no", "harmful", "hate", "hateful"],
    },
    "MMHS": {
        "task_key": "MMHS150K::hate_category::label",
        "tokens": ["yes", "no", "racist", "Racist", "sexist", "Sexist",
                   "offensive", "hate", "hateful", "gay", "Jewish", "Muslim", "Black"],
    },
    "Crisis_A": {
        "task_key": "Crisis_HateMM::subtask_a_hate_speech::label",
        "tokens": ["yes", "no", "hate", "hateful", "offensive"],
    },
}


@dataclass
class RoutingFeatureConfig:
    jlens_matrix_path: str = ""       # jlens_matrix_layer22.safetensors
    target_token_ids_path: str = ""   # target_token_ids.json
    residual_sae_path: str = ""       # dir with sae.safetensors + residual_stats.json
    probe_joblib_dir: str = ""        # dir with probe_<task>.joblib files (for cross-reference)
    output_dir: str = ""
    top_k: int = 20                   # routing features to select per task (matches silent top_k)
    top_k_probe_for_overlap: int = 20 # probe top-K to compare against for overlap/jaccard
    tasks: list[str] = field(default_factory=lambda: ["MAMI", "HarMeme", "MMHS", "Crisis_A"])


def _read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """Minimal safetensors reader (avoids a hard dependency if unavailable)."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
        raw = f.read()
    dtype_map = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    tensors = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        shape = meta["shape"]
        dtype = dtype_map[meta["dtype"]]
        start, end = meta["data_offsets"]
        buf = raw[start:end]
        t = torch.frombuffer(bytearray(buf), dtype=dtype).reshape(shape)
        tensors[key] = t
    return tensors


def _load_probe_top_features(joblib_dir: Path, task_key: str, top_k: int) -> tuple[list[int], dict[int, float]]:
    task_safe = task_key.replace("::", "__").replace(" ", "_").replace("/", "_")
    candidates = list(joblib_dir.glob(f"probe_{task_safe}*.joblib"))
    if not candidates:
        short = task_key.split("::")[0] + "__" + task_key.split("::")[1]
        candidates = list(joblib_dir.glob(f"probe_{short}*.joblib"))
    if not candidates:
        logger.warning("No probe joblib found for %s in %s", task_key, joblib_dir)
        return [], {}
    bundle = joblib.load(candidates[0])
    coef = bundle["clf"].coef_
    if coef.ndim == 2 and coef.shape[0] == 1:
        coef = coef[0]
    elif coef.ndim == 2:
        coef = coef[abs(coef).max(axis=1).argmax()]
    top_idx = (-abs(coef)).argsort()[:top_k]
    weight_by_idx = {int(i): float(coef[i]) for i in range(len(coef))}
    return [int(i) for i in top_idx], weight_by_idx


def run(cfg: RoutingFeatureConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cached J_avg[layer22] matrix + target token ids ──────────────────
    jlens_tensors = _read_safetensors(Path(cfg.jlens_matrix_path))
    j_key = next(k for k in jlens_tensors if k.startswith("jlens_"))
    j_avg = jlens_tensors[j_key].float()  # [n_target, d_model]
    token_ids: dict[str, int] = json.loads(Path(cfg.target_token_ids_path).read_text())
    token_names = list(token_ids.keys())
    assert j_avg.shape[0] == len(token_names), (
        f"J_avg has {j_avg.shape[0]} rows but target_token_ids.json has {len(token_names)} tokens"
    )
    name_to_row = {name: i for i, name in enumerate(token_names)}
    logger.info("Loaded J_avg[%s]: shape=%s, tokens=%s", j_key, tuple(j_avg.shape), token_names)

    # ── Load residual SAE decoder + normalization stats ────────────────────────
    sae_tensors = _read_safetensors(Path(cfg.residual_sae_path) / "sae.safetensors")
    w_dec = sae_tensors["W_dec"].float()  # [num_latents, d_model]
    stats = json.loads((Path(cfg.residual_sae_path) / "residual_stats.json").read_text())
    std = torch.tensor(stats["std"], dtype=torch.float32)  # [d_model]
    assert w_dec.shape[1] == j_avg.shape[1] == std.shape[0], (
        f"d_model mismatch: W_dec={w_dec.shape[1]}, J_avg={j_avg.shape[1]}, std={std.shape[0]}"
    )
    num_latents = w_dec.shape[0]
    logger.info("Residual SAE: num_latents=%d, d_model=%d", num_latents, w_dec.shape[1])

    # J_avg scaled by std, so jspace_scores = J_avg_scaled @ W_dec.T directly
    # accounts for the SAE's denormalize() step (see module docstring).
    j_avg_scaled = j_avg * std[None, :]           # [n_target, d_model]
    jspace_scores = j_avg_scaled @ w_dec.T         # [n_target, num_latents]
    jspace_scores = jspace_scores.numpy()

    probe_joblib_dir = Path(cfg.probe_joblib_dir)
    summary: dict[str, Any] = {}

    for task in cfg.tasks:
        meta = TASK_TOKEN_MAP.get(task)
        assert meta, f"Unknown task {task!r}; choose from {list(TASK_TOKEN_MAP)}"
        rows = [name_to_row[t] for t in meta["tokens"] if t in name_to_row]
        missing = [t for t in meta["tokens"] if t not in name_to_row]
        if missing:
            logger.warning("Task %s: tokens not in J_avg vocabulary: %s", task, missing)
        assert rows, f"No usable target tokens found for task {task!r}"

        # Per-feature routing score: max |jspace_score| over this task's relevant tokens
        task_scores = np.abs(jspace_scores[rows, :]).max(axis=0)  # [num_latents]
        routing_top_idx = (-task_scores).argsort()[: cfg.top_k]

        probe_top_idx, probe_weight_by_idx = _load_probe_top_features(
            probe_joblib_dir, meta["task_key"], cfg.top_k_probe_for_overlap)

        routing_set = set(int(i) for i in routing_top_idx)
        probe_set = set(probe_top_idx)
        overlap = routing_set & probe_set
        jaccard = len(overlap) / len(routing_set | probe_set) if (routing_set | probe_set) else 0.0

        task_summary = {
            "task_key": meta["task_key"],
            "tokens_used": [meta["tokens"][i] for i in range(len(meta["tokens"])) if meta["tokens"][i] in name_to_row],
            "routing_top_features": [int(i) for i in routing_top_idx],
            "routing_top_scores": [round(float(task_scores[i]), 6) for i in routing_top_idx],
            "probe_top_features": probe_top_idx,
            "overlap_routing_and_probe": sorted(overlap),
            "n_overlap": len(overlap),
            "jaccard_routing_vs_probe": round(jaccard, 4),
        }
        summary[task] = task_summary
        logger.info(
            "%s: routing top-%d vs probe top-%d overlap=%d/%d (jaccard=%.3f)",
            task, cfg.top_k, cfg.top_k_probe_for_overlap, len(overlap), len(routing_set | probe_set), jaccard,
        )

    (out_dir / "routing_features_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote routing_features_summary.json to %s", out_dir)

    print("\n" + "=" * 80)
    print("Residual-SAE Routing Features vs Probe (Silent) Features")
    print(f"{'Task':10s}  {'RoutingTop':>10}  {'ProbeTop':>10}  {'Overlap':>8}  {'Jaccard':>8}")
    print("-" * 80)
    for task, s in summary.items():
        print(f"{task:10s}  {cfg.top_k:>10}  {len(s['probe_top_features']):>10}  "
              f"{s['n_overlap']:>8}  {s['jaccard_routing_vs_probe']:>8.3f}")
    print("=" * 80)
    print("Low overlap/jaccard confirms routing and silent (probe) feature sets are largely")
    print("disjoint within the residual SAE dictionary, mirroring the FHM crosscoder finding")
    print("(routing features f6786/f5536/f5449/f5353 vs silent probe features f10445/f15987).")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(RoutingFeatureConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
