"""Evaluate a dedicated MMHS adapter on separate binary and subtype metrics."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.train_lora_distillation_mmhs import _validation_samples, evaluate
from gemma3_experiments.residual_sae import GemmaResidualConfig, load_gemma3_model, resolve_visible_device


@dataclass
class EvaluateLoraDistillationMMHSConfig:
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    dataset_root: str = "./data"
    lora_adapter_path: str = ""
    validation_per_class_cap: int = 10_000
    output_path: str = "./lora_distillation_mmhs_eval.json"


def run(cfg: EvaluateLoraDistillationMMHSConfig) -> None:
    from peft import PeftModel

    device = resolve_visible_device(cfg.gemma.model_device)
    model, processor = load_gemma3_model(cfg.gemma)
    if cfg.lora_adapter_path:
        model = PeftModel.from_pretrained(model, cfg.lora_adapter_path).merge_and_unload()
    model.eval()
    metrics = evaluate(model, processor, device, _validation_samples(cfg.dataset_root, cfg.validation_per_class_cap))
    Path(cfg.output_path).write_text(json.dumps(metrics, indent=2))
    logging.getLogger(__name__).info("Wrote MMHS binary/subtype evaluation to %s", cfg.output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse(EvaluateLoraDistillationMMHSConfig, argument_generation_mode=ArgumentGenerationMode.BOTH))


if __name__ == "__main__":
    main()