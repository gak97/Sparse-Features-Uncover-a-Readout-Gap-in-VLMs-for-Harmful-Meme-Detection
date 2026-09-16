"""Evaluate the pilot residual-SAE reconstruction hook on locked MultiBully."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from simple_parsing import Serializable, list_field, parse

from gemma3_experiments.residual_sae.normalization import load_residual_stats

from ..config import QwenResidualConfig
from ..joint import make_joint_reconstruction_hook
from ..models import load_residual_sae
from ..qwen import QwenSae, classify_constrained, load_qwen3_model, resolve_target_module
from .analyze_exist_multilingual import _read_jsonl
from .analyze_multibully import _metrics


@dataclass
class MultiBullyHookConfig(Serializable):
    cache_dir: str = "./outputs/multibully/dense_raw"
    output_dir: str = "./outputs/multibully/hook_raw_ocr"
    residual_sae_path: str = "./outputs/qwen_residual_sae_ckpts/best_finite"
    alphas: list[float] = list_field(0.25, 0.35)
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    device: str = "cuda:0"
    hook_module_path: str = "model.language_model.layers.20"
    resume: bool = True


def _path(output_dir: Path, alpha: float) -> Path:
    return output_dir / f"hook_alpha_{alpha:.2f}.jsonl"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(MultiBullyHookConfig)
    cache_dir, output_dir = Path(cfg.cache_dir), Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = [
        row for row in _read_jsonl(cache_dir / "manifest.jsonl")
        if row.get("split") == "eval" and row.get("task_name") == "cyberbullying_detection"
    ]
    if len(evaluation) != 1000:
        raise ValueError(f"Expected locked 1,000-example evaluation set, found {len(evaluation)}")
    qwen_cfg = QwenResidualConfig(model_id=cfg.model_id, model_device=cfg.device, hook_module_path=cfg.hook_module_path)
    model, processor = load_qwen3_model(qwen_cfg)
    target_module = resolve_target_module(model, qwen_cfg)
    base_sae = QwenSae.load_from_hub(qwen_cfg.sae_repo_id, qwen_cfg.sae_layer, device=cfg.device)
    residual_sae = load_residual_sae(cfg.residual_sae_path, device=cfg.device)
    residual_stats = load_residual_stats(cfg.residual_sae_path)

    summaries: dict[str, dict] = {}
    for alpha in cfg.alphas:
        output_path = _path(output_dir, alpha)
        completed = {json.loads(line)["sample_id"] for line in output_path.open()} if cfg.resume and output_path.exists() else set()
        remaining = [row for row in evaluation if row["sample_id"] not in completed]
        hook = make_joint_reconstruction_hook(base_sae, residual_sae, reconstruction_alpha=alpha, residual_stats=residual_stats)
        with output_path.open("a") as fd:
            for index, row in enumerate(remaining, start=1):
                image = Image.open(row["image_path"]).convert("RGB")
                handle = target_module.register_forward_hook(hook)
                try:
                    predicted, text, confidence = classify_constrained(model, processor, image, row["prompt_text"], cfg.device)
                finally:
                    handle.remove()
                fd.write(json.dumps({
                    "sample_id": row["sample_id"],
                    "gold": row["gold_fields"]["label"],
                    "prediction": "bully" if predicted else "non-bully",
                    "prediction_text": text,
                    "conf_gap": confidence,
                    "reconstruction_alpha": alpha,
                }) + "\n")
                fd.flush()
                if index % 25 == 0 or index == len(remaining):
                    logging.info("Hook alpha=%.2f progress: %d/%d", alpha, index, len(remaining))
        rows = _read_jsonl(output_path)
        gold = np.asarray([int(row["gold"] == "bully") for row in rows], dtype=np.int32)
        prediction = np.asarray([int(row["prediction"] == "bully") for row in rows], dtype=np.int32)
        summaries[f"alpha_{alpha:.2f}"] = {"n_eval": len(rows), **_metrics(gold, prediction)}
    (output_dir / "hook_summary.json").write_text(json.dumps({"residual_sae_path": cfg.residual_sae_path, "summaries": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
