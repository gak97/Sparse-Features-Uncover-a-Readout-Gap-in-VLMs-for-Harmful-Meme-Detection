

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
from .analyze_exist_multilingual import _metrics, _read_jsonl

logger = logging.getLogger(__name__)


@dataclass
class ExistHookConfig(Serializable):
    cache_dir: str = "./outputs/exist_multilingual/dense_raw"
    output_dir: str = "./outputs/exist_multilingual/hook_raw_ocr"
    residual_sae_path: str = "./outputs/qwen_residual_sae_ckpts/best_finite"
    alphas: list[float] = list_field(0.25, 0.35)
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    device: str = "cuda:0"
    hook_module_path: str = "model.language_model.layers.20"
    resume: bool = True


def _output_path(output_dir: Path, alpha: float) -> Path:
    return output_dir / f"hook_alpha_{alpha:.2f}.jsonl"


def _summarize(rows: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for language in ("en", "es", "all"):
        subset = rows if language == "all" else [row for row in rows if row["language"] == language]
        gold = np.asarray([int(row["gold"] == "sexist") for row in subset], dtype=np.int32)
        prediction = np.asarray([int(row["prediction"] == "sexist") for row in subset], dtype=np.int32)
        output[language] = {"n_eval": len(subset), **_metrics(gold, prediction)}
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ExistHookConfig)
    cache_dir, output_dir = Path(cfg.cache_dir), Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = [
        row for row in _read_jsonl(cache_dir / "manifest.jsonl")
        if row.get("split") == "eval" and row.get("task_name") == "sexism_detection"
    ]
    if len(eval_rows) != 500:
        raise ValueError(f"Expected the fixed 500-sample EXIST evaluation set, found {len(eval_rows)} rows")

    qwen_cfg = QwenResidualConfig(model_id=cfg.model_id, model_device=cfg.device, hook_module_path=cfg.hook_module_path)
    model, processor = load_qwen3_model(qwen_cfg)
    target_module = resolve_target_module(model, qwen_cfg)
    base_sae = QwenSae.load_from_hub(qwen_cfg.sae_repo_id, qwen_cfg.sae_layer, device=cfg.device)
    residual_sae = load_residual_sae(cfg.residual_sae_path, device=cfg.device)
    residual_stats = load_residual_stats(cfg.residual_sae_path)

    summaries: dict[str, dict] = {}
    for alpha in cfg.alphas:
        path = _output_path(output_dir, alpha)
        completed = {json.loads(line)["sample_id"] for line in path.open()} if cfg.resume and path.exists() else set()
        remaining = [row for row in eval_rows if row["sample_id"] not in completed]
        hook = make_joint_reconstruction_hook(
            base_sae,
            residual_sae,
            reconstruction_alpha=alpha,
            residual_stats=residual_stats,
        )
        logger.info("Hook alpha=%.2f: %d already complete, %d remaining", alpha, len(completed), len(remaining))
        with path.open("a") as fd:
            for index, row in enumerate(remaining, start=1):
                image = Image.open(row["image_path"]).convert("RGB")
                handle = target_module.register_forward_hook(hook)
                try:
                    predicted, text, confidence = classify_constrained(model, processor, image, row["prompt_text"], cfg.device)
                finally:
                    handle.remove()
                fd.write(json.dumps({
                    "sample_id": row["sample_id"],
                    "language": row.get("raw_fields", {}).get("language"),
                    "gold": row["gold_fields"]["label"],
                    "prediction": "sexist" if predicted else "non-sexist",
                    "prediction_text": text,
                    "conf_gap": confidence,
                    "reconstruction_alpha": alpha,
                    "residual_sae_path": cfg.residual_sae_path,
                }) + "\n")
                fd.flush()
                if index % 25 == 0 or index == len(remaining):
                    logger.info("Hook alpha=%.2f progress: %d/%d", alpha, index, len(remaining))
        summaries[f"alpha_{alpha:.2f}"] = _summarize(_read_jsonl(path))

    (output_dir / "hook_summary.json").write_text(json.dumps({
        "residual_sae_path": cfg.residual_sae_path,
        "summaries": summaries,
    }, indent=2) + "\n")
    logger.info("Wrote EXIST hook results to %s", output_dir)


if __name__ == "__main__":
    main()
