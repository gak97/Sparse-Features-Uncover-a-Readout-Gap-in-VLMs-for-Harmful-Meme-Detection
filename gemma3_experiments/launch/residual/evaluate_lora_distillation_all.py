

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.generate_lora_distillation_targets_all import TASK_SPECS
from gemma3_experiments.launch.residual.probe_validate import _per_class_metrics
from gemma3_experiments.residual_sae import (
    GemmaResidualConfig,
    MemeDatasetConfig,
    classify_constrained,
    load_gemma3_model,
    load_meme_samples,
    resolve_visible_device,
    score_label_candidates,
)
from gemma3_experiments.residual_sae.datasets import build_primary_label_prompt, build_simple_classify_prompt

logger = logging.getLogger(__name__)


@dataclass
class EvaluateLoraDistillationAllConfig:
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    lora_adapter_path: str = ""   # empty = no-LoRA base-LVLM baseline
    task_keys: list[str] = field(default_factory=list)  # empty = all TASK_SPECS
    max_samples_per_task: int | None = None
    output_dir: str = "./lora_distillation_eval_all"


def _evaluate_task(spec: dict[str, Any], model: Any, processor: Any, device: str, max_samples: int | None) -> dict[str, Any]:
    dataset_name = spec["dataset_name"]
    task_name = spec["task_name"]
    labels = spec["labels"]

    samples = [
        s for s in load_meme_samples(MemeDatasetConfig(datasets=[dataset_name], splits=["validation"]))
        if s.task_name == task_name
    ]
    if max_samples is not None:
        samples = samples[:max_samples]
    assert samples, f"No validation samples found for {dataset_name}/{task_name}"

    gold_labels: list[str] = []
    pred_labels: list[str] = []
    rows: list[dict[str, Any]] = []
    for sample in samples:
        gold = sample.gold_fields.get(sample.schema.primary_field.name)
        if gold is None:
            continue
        image = Image.open(sample.image_path).convert("RGB")

        if spec["decode_strategy"] == "yes_no":
            prompt_text = build_simple_classify_prompt(sample)
            is_positive, _, conf = classify_constrained(model, processor, image, prompt_text, device)
            pred = spec["yes_no_positive_label"] if is_positive else next(lbl for lbl in labels if lbl != spec["yes_no_positive_label"])
        else:
            prompt_text = build_primary_label_prompt(sample)
            pred, _, conf = score_label_candidates(model, processor, image, prompt_text, list(labels), device)

        gold_labels.append(gold)
        pred_labels.append(pred)
        rows.append({"sample_id": sample.sample_id, "gold": gold, "pred": pred, "confidence": round(conf, 4)})

    assert gold_labels, f"No gold labels found for {dataset_name}/{task_name}"
    metrics = _per_class_metrics(gold_labels, pred_labels)
    return {"metrics": metrics, "rows": rows}


def run(cfg: EvaluateLoraDistillationAllConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_visible_device(cfg.gemma.model_device)
    model, processor = load_gemma3_model(cfg.gemma)
    if cfg.lora_adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.lora_adapter_path)
        model = model.merge_and_unload()
        logger.info("Loaded + merged LoRA adapter from %s", cfg.lora_adapter_path)
    else:
        logger.info("No lora_adapter_path given -- running the no-LoRA base-LVLM baseline")
    model.eval()

    specs = TASK_SPECS
    if cfg.task_keys:
        wanted = set(cfg.task_keys)
        specs = [s for s in TASK_SPECS if f"{s['dataset_name']}::{s['task_name']}::{s['field_name']}" in wanted]
        assert specs, f"No TASK_SPECS matched task_keys={cfg.task_keys}"

    all_results: dict[str, Any] = {}
    macro_f1s: list[float] = []
    for spec in specs:
        task_key = f"{spec['dataset_name']}::{spec['task_name']}::{spec['field_name']}"
        logger.info("=== Evaluating %s ===", task_key)
        result = _evaluate_task(spec, model, processor, device, cfg.max_samples_per_task)
        all_results[task_key] = result["metrics"]
        macro_f1s.append(result["metrics"]["macro_f1"])
        with (out_dir / f"predictions_{task_key.replace('::', '__').replace(' ', '_')}.jsonl").open("w") as fd:
            for row in result["rows"]:
                fd.write(json.dumps(row) + "\n")
        logger.info(
            "%s: n=%d macro_f1=%.4f accuracy=%.4f",
            task_key, result["metrics"]["n"], result["metrics"]["macro_f1"], result["metrics"]["accuracy"],
        )

    overall = {
        "mode": f"LoRA adapter: {cfg.lora_adapter_path}" if cfg.lora_adapter_path else "no-LoRA base-LVLM baseline",
        "macro_f1_averaged_across_tasks": round(sum(macro_f1s) / len(macro_f1s), 4) if macro_f1s else None,
        "per_task_metrics": all_results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(overall, indent=2))
    logger.info("Wrote metrics.json to %s", out_dir)

    print("\n" + "=" * 88)
    print(f"Stage 3 (comprehensive) evaluation -- {overall['mode']}")
    print(f"macro_f1 averaged across {len(macro_f1s)} tasks: {overall['macro_f1_averaged_across_tasks']}")
    print(f"{'task_key':60s} {'n':>6s} {'macro_f1':>10s} {'accuracy':>10s}")
    for task_key, metrics in all_results.items():
        print(f"{task_key:60s} {metrics['n']:>6d} {metrics['macro_f1']:>10.4f} {metrics['accuracy']:>10.4f}")
    print("=" * 88)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(EvaluateLoraDistillationAllConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
