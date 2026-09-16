

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.probe_validate import _per_class_metrics
from gemma3_experiments.residual_sae import (
    GemmaResidualConfig,
    MemeDatasetConfig,
    classify_constrained,
    load_gemma3_model,
    load_meme_samples,
    resolve_visible_device,
)
from gemma3_experiments.residual_sae.datasets import build_simple_classify_prompt

logger = logging.getLogger(__name__)


@dataclass
class EvaluateLoraDistillationConfig:
    dataset: MemeDatasetConfig = field(
        default_factory=lambda: MemeDatasetConfig(datasets=["Hateful_Memes_Dataset"], splits=["validation"])
    )
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    lora_adapter_path: str = ""   # empty string = no-LoRA base-LVLM baseline
    max_samples: int | None = None
    output_dir: str = "./lora_distillation_eval"


def run(cfg: EvaluateLoraDistillationConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_visible_device(cfg.gemma.model_device)

    samples = load_meme_samples(cfg.dataset)
    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]
    logger.info("Loaded %d samples", len(samples))

    model, processor = load_gemma3_model(cfg.gemma)
    if cfg.lora_adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.lora_adapter_path)
        model = model.merge_and_unload()
        logger.info("Loaded + merged LoRA adapter from %s", cfg.lora_adapter_path)
    else:
        logger.info("No lora_adapter_path given -- running the no-LoRA base-LVLM baseline")
    model.eval()

    rows: list[dict[str, Any]] = []
    gold_labels: list[str] = []
    pred_labels: list[str] = []
    log_every = max(1, len(samples) // 20)
    for i, sample in enumerate(samples, start=1):
        gold = sample.gold_fields.get(sample.schema.primary_field.name)
        if gold is None:
            continue
        prompt_text = build_simple_classify_prompt(sample)
        image = Image.open(sample.image_path).convert("RGB")
        is_positive, _, conf_gap = classify_constrained(model, processor, image, prompt_text, device)
        primary_labels = sample.schema.primary_field.labels
        pred_label = primary_labels[-1] if is_positive else primary_labels[0]
        rows.append({
            "sample_id": sample.sample_id,
            "gold": gold,
            "pred": pred_label,
            "conf_gap": round(conf_gap, 4),
            "correct": pred_label == gold,
        })
        gold_labels.append(gold)
        pred_labels.append(pred_label)
        if i % log_every == 0:
            logger.info("Processed %d/%d", i, len(samples))

    assert gold_labels, "No samples with a gold label were found -- check dataset/split"
    metrics = _per_class_metrics(gold_labels, pred_labels)

    with (out_dir / "predictions.jsonl").open("w") as fd:
        for row in rows:
            fd.write(json.dumps(row) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote predictions.jsonl + metrics.json to %s", out_dir)

    print("\n" + "=" * 72)
    mode = f"LoRA adapter: {cfg.lora_adapter_path}" if cfg.lora_adapter_path else "no-LoRA base-LVLM baseline"
    print(f"Stage 3 evaluation -- {mode}")
    print(f"n={metrics['n']}  macro_f1={metrics['macro_f1']}  accuracy={metrics['accuracy']}")
    print(f"pred_counts={metrics['pred_counts']}  gold_counts={metrics['gold_counts']}")
    print("=" * 72)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(EvaluateLoraDistillationConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
