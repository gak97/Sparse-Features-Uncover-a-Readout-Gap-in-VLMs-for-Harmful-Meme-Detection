

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.probe_validate import _per_class_metrics
from gemma3_experiments.meme_safety_probe import MemeSafetyProbe, MemeSafetyProbeConfig
from gemma3_experiments.residual_sae import MemeDatasetConfig, load_meme_samples

logger = logging.getLogger(__name__)


@dataclass
class ValidateWrapperConfig:
    dataset: MemeDatasetConfig = field(
        default_factory=lambda: MemeDatasetConfig(datasets=["Hateful_Memes_Dataset"], splits=["validation"])
    )
    probe: MemeSafetyProbeConfig = field(default_factory=MemeSafetyProbeConfig)

    # probe_val_metrics.json from the SAME probe_dir, used purely as the ground
    # truth to diff against -- not loaded/used by the wrapper itself.
    reference_metrics_path: str = ""
    reference_task_key: str = ""  # e.g. "Hateful_Memes_Dataset::hateful_memes_binary::label"

    max_samples: int | None = None
    output_dir: str = "./meme_safety_probe_wrapper_validation"


def run(cfg: ValidateWrapperConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_meme_samples(cfg.dataset)
    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]
    logger.info("Loaded %d samples", len(samples))

    probe = MemeSafetyProbe(cfg.probe)

    rows: list[dict[str, Any]] = []
    gold_labels: list[str] = []
    pred_labels: list[str] = []
    log_every = max(1, len(samples) // 20)
    for i, sample in enumerate(samples, start=1):
        gold = sample.gold_fields.get(sample.schema.primary_field.name)
        if gold is None:
            continue
        result = probe.predict(sample.image_path, sample.meme_text, schema=sample.schema)
        rows.append({
            "sample_id": sample.sample_id,
            "gold": gold,
            "pred": result.label,
            "confidence": result.confidence,
            "label_scores": result.label_scores,
            "correct": result.label == gold,
        })
        gold_labels.append(gold)
        pred_labels.append(result.label)
        if i % log_every == 0:
            logger.info("Processed %d/%d", i, len(samples))

    assert gold_labels, "No samples with a gold label were found -- check dataset/split"
    metrics = _per_class_metrics(gold_labels, pred_labels)

    with (out_dir / "wrapper_predictions.jsonl").open("w") as fd:
        for row in rows:
            fd.write(json.dumps(row) + "\n")
    (out_dir / "wrapper_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote wrapper_predictions.jsonl + wrapper_metrics.json to %s", out_dir)

    print("\n" + "=" * 72)
    print(f"MemeSafetyProbe wrapper validation -- n={metrics['n']}")
    print(f"macro_f1={metrics['macro_f1']}  accuracy={metrics['accuracy']}")
    stage_note = "Stage 1 (SAE features)" if cfg.probe.use_residual_sae else "Stage 2 (raw hidden-state features)"
    calibration_note = (
        "CALIBRATED (positive_label + calibrated_threshold applied)"
        if cfg.probe.calibrated_threshold is not None else "RAW (uncalibrated .predict())"
    )
    print(f"mode: {stage_note}, {calibration_note}")

    if cfg.reference_metrics_path and cfg.reference_task_key:
        ref = json.loads(Path(cfg.reference_metrics_path).read_text())
        ref_task = ref.get(cfg.reference_task_key, {})
        if cfg.probe.calibrated_threshold is not None:
            ref_block = "calibrated_metrics"
        elif not cfg.probe.use_residual_sae:
            ref_block = "raw_residual_control"
        else:
            ref_block = "probe_metrics"
        ref_f1 = ref_task.get(ref_block, {}).get("macro_f1")
        print(f"reference (probe_validate.py, cache-based, {ref_block}) macro_f1={ref_f1}")
        if ref_f1 is not None:
            delta = round(metrics["macro_f1"] - ref_f1, 4)
            verdict = "MATCH (release gate PASSED)" if abs(delta) <= 0.02 else "MISMATCH (release gate FAILED)"
            print(f"delta (wrapper - reference) = {delta}  ->  {verdict}")
    print("=" * 72)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ValidateWrapperConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
