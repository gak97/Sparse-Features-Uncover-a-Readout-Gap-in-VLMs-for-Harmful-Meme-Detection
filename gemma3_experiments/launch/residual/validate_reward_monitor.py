

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.probe_validate import _per_class_metrics, _read_jsonl
from gemma3_experiments.launch.residual.reward_monitor import RewardMonitor, RewardMonitorConfig, RewardWeights
from gemma3_experiments.meme_safety_probe.wrapper import HATEFUL_MEMES_SCHEMA
from gemma3_experiments.residual_sae import MemeDatasetConfig, load_meme_samples
from gemma3_experiments.residual_sae.datasets import MAMI_SCHEMA

logger = logging.getLogger(__name__)

_SCHEMA_BY_TASK = {
    "hateful_memes_binary": HATEFUL_MEMES_SCHEMA,
    "misogyny_and_subtypes": MAMI_SCHEMA,
}


@dataclass
class ValidateRewardMonitorConfig:
    reward_monitor: RewardMonitorConfig = field(default_factory=RewardMonitorConfig)
    # Held-out split ONLY -- see module docstring. Defaults intentionally do
    # NOT include "train" anywhere.
    dataset: MemeDatasetConfig = field(
        default_factory=lambda: MemeDatasetConfig(datasets=["Hateful_Memes_Dataset"], splits=["test"])
    )
    identity_group_controls_path: str = ""       # from build_identity_group_controls.py (held-out split)
    dataset_name: str = "Hateful_Memes_Dataset"  # filters controls to THIS run's probe/schema
    task_name: str = "hateful_memes_binary"
    identity_group_fp_threshold: float = 0.3     # hard-gate threshold, see module docstring
    max_samples: int | None = None
    output_dir: str = "./reward_monitor_validation"


def run(cfg: ValidateRewardMonitorConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = _SCHEMA_BY_TASK.get(cfg.task_name, HATEFUL_MEMES_SCHEMA)
    assert "train" not in cfg.dataset.splits, (
        "Refusing to validate the reward monitor against a TRAIN split -- "
        "the SAE probe was fit on train data, so this would be a circular, "
        "misleadingly inflated check. Use --dataset.splits test (or "
        "validation if no test split exists / is too small)."
    )

    monitor = RewardMonitor(cfg.reward_monitor, RewardWeights())

    result: dict[str, Any] = {}

    # ── Check 1: reward accuracy on a genuinely held-out split ─────────────
    samples = load_meme_samples(cfg.dataset)
    if cfg.max_samples is not None:
        samples = samples[: cfg.max_samples]
    assert samples, f"No samples found for {cfg.dataset.datasets}/{cfg.dataset.splits}"
    gold_labels: list[str] = []
    pred_labels: list[str] = []
    for i, sample in enumerate(samples, start=1):
        gold = sample.gold_fields.get(sample.schema.primary_field.name)
        if gold is None:
            continue
        assessment = monitor.assess(sample.image_path, sample.meme_text, schema=schema)
        gold_labels.append(gold)
        pred_labels.append(assessment.prediction.label)
        if i % max(1, len(samples) // 20) == 0:
            logger.info("Reward-accuracy check: %d/%d", i, len(samples))
    assert gold_labels, "No labeled samples found -- for FHM test, did you pass --dataset.fhm_labels_path / --dataset.fhm_label_split?"
    result["reward_accuracy"] = _per_class_metrics(gold_labels, pred_labels)
    result["reward_accuracy_split"] = list(cfg.dataset.splits)
    logger.info("Reward-accuracy macro_f1=%.4f (split=%s)", result["reward_accuracy"]["macro_f1"], cfg.dataset.splits)

    # ── Check 2: false positives on benign identity-group controls ────────
    if cfg.identity_group_controls_path:
        all_controls = _read_jsonl(Path(cfg.identity_group_controls_path))
        # IMPORTANT: only score controls from THIS run's own dataset -- the
        # loaded probe/schema is task-specific, so scoring e.g. MAMI controls
        # under an FHM probe would silently under-report false positives (the
        # FHM classifier never predicts MAMI's "misogynous" label at all).
        controls = [c for c in all_controls if c["dataset_name"] == cfg.dataset_name]
        assert controls, (
            f"No controls found for dataset_name={cfg.dataset_name!r} in {cfg.identity_group_controls_path} "
            f"(found datasets: {sorted({c['dataset_name'] for c in all_controls})})"
        )
        n_fp = 0
        n_total = 0
        fp_rows: list[dict[str, Any]] = []
        for control in controls:
            assessment = monitor.assess(control["image_path"], control["meme_text"], schema=schema)
            n_total += 1
            is_fp = assessment.prediction.label == control["positive_label"]
            n_fp += int(is_fp)
            if is_fp:
                fp_rows.append({
                    "sample_id": control["sample_id"],
                    "dataset_name": control["dataset_name"],
                    "matched_term": control["matched_term"],
                    "predicted_label": assessment.prediction.label,
                    "confidence": assessment.prediction.confidence,
                })
        fp_rate = n_fp / n_total if n_total else 0.0
        result["identity_group_false_positive_rate"] = round(fp_rate, 4)
        result["identity_group_n_controls"] = n_total
        result["identity_group_n_false_positives"] = n_fp
        (out_dir / "identity_group_false_positives.jsonl").write_text(
            "\n".join(json.dumps(row) for row in fp_rows) + ("\n" if fp_rows else "")
        )
        logger.info("Identity-group control false-positive rate=%.4f (%d/%d)", fp_rate, n_fp, n_total)
        result["gate_passed"] = fp_rate <= cfg.identity_group_fp_threshold
        result["identity_group_fp_threshold"] = cfg.identity_group_fp_threshold
        if not result["gate_passed"]:
            logger.error(
                "GATE FAILED: identity-group FP rate %.4f exceeds threshold %.4f -- "
                "do NOT use this reward monitor for train_rlfr.py until fixed.",
                fp_rate, cfg.identity_group_fp_threshold,
            )

    (out_dir / "reward_monitor_validation.json").write_text(json.dumps(result, indent=2))
    logger.info("Wrote reward_monitor_validation.json to %s", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(ValidateRewardMonitorConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
