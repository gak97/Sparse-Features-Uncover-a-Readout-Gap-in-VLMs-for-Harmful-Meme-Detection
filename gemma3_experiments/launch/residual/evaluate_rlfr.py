

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.probe_validate import _aggregate_sample, _per_class_metrics, _read_jsonl
from gemma3_experiments.launch.residual.reward_monitor import RewardMonitor, RewardMonitorConfig, RewardWeights
from gemma3_experiments.meme_safety_probe.wrapper import HATEFUL_MEMES_SCHEMA
from gemma3_experiments.residual_sae import (
    GemmaResidualConfig,
    GemmaSae,
    MemeDatasetConfig,
    MemeSample,
    TaskSchema,
    capture_task_hidden_states,
    classify_constrained,
    load_gemma3_model,
    load_meme_samples,
    load_residual_sae,
    load_residual_stats,
    resolve_visible_device,
    run_joint_saes,
)
from gemma3_experiments.residual_sae.cache import _token_roles
from gemma3_experiments.residual_sae.datasets import MAMI_SCHEMA, build_simple_classify_prompt

logger = logging.getLogger(__name__)

_SCHEMA_BY_TASK = {
    "hateful_memes_binary": HATEFUL_MEMES_SCHEMA,
    "misogyny_and_subtypes": MAMI_SCHEMA,
}


@dataclass
class EvaluateRLFRConfig:
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    dataset: MemeDatasetConfig = field(
        default_factory=lambda: MemeDatasetConfig(datasets=["Hateful_Memes_Dataset"], splits=["test"])
    )
    task_name: str = "hateful_memes_binary"
    dataset_name: str = "Hateful_Memes_Dataset"
    positive_label: str = "hateful"
    negative_label: str = "not hateful"

    sft_adapter_path: str = ""                # Stage 3 SFT LoRA (empty = skip this comparison arm)
    rlfr_adapter_path: str = ""                # Stage 4 RLFR LoRA (empty = skip this comparison arm)

    probe: RewardMonitorConfig = field(default_factory=RewardMonitorConfig)  # Stage 1 probe wrapper for the paired comparison
    identity_group_controls_path: str = ""

    residual_sae_path: str = ""               # for the Step 6 reward-hacking feature-overlap check
    reward_hacking_n_samples: int = 40
    reward_hacking_top_k: int = 20
    reward_hacking_token_selection: str = "all"
    reward_hacking_aggregation: str = "max"

    human_audit_n_examples: int = 20
    max_samples: int | None = None
    output_dir: str = "./rlfr_eval"


def _load_variant(cfg: EvaluateRLFRConfig, adapter_path: str, device: str):
    model, processor = load_gemma3_model(cfg.gemma)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.info("Loaded + merged adapter from %s", adapter_path)
    model.eval()
    return model, processor


def _rec_fields(rec: Any) -> tuple[str, str, str]:
    """Accessor shared by real MemeSample eval rows and dict-based identity-
    group control rows (from build_identity_group_controls.py's jsonl)."""
    if isinstance(rec, MemeSample):
        return rec.sample_id, rec.image_path, rec.meme_text
    return rec["sample_id"], rec["image_path"], rec["meme_text"]


def _run_classification(model, processor, device, samples: list[Any], schema: TaskSchema) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        sample_id, image_path, meme_text = _rec_fields(sample)
        m = MemeSample(
            dataset_name="eval", task_name=schema.task_name, sample_id=sample_id, split="eval",
            image_path=image_path, meme_text=meme_text, schema=schema,
            gold_fields={}, raw_fields={},
        )
        prompt_text = build_simple_classify_prompt(m)
        image = Image.open(image_path).convert("RGB")
        is_positive, _, conf_gap = classify_constrained(model, processor, image, prompt_text, device)
        primary_labels = schema.primary_field.labels
        pred_label = primary_labels[-1] if is_positive else primary_labels[0]
        rows.append({"sample_id": sample_id, "pred": pred_label, "conf_gap": round(conf_gap, 4)})
    return rows


def _extract_feature_vector(model, processor, device, sample: Any, schema: TaskSchema, gemma_cfg: GemmaResidualConfig,
                             base_sae, residual_sae, residual_stats, token_selection: str, aggregation: str) -> torch.Tensor | None:
    sample_id, image_path, meme_text = _rec_fields(sample)
    m = MemeSample(
        dataset_name="eval", task_name=schema.task_name, sample_id=sample_id, split="eval",
        image_path=image_path, meme_text=meme_text, schema=schema, gold_fields={}, raw_fields={},
    )
    run = capture_task_hidden_states(model, processor, m, gemma_cfg)
    roles = _token_roles(run.hidden.shape[0], run.image_span, run.input_len)
    hidden = run.hidden.to(device=device, dtype=torch.float32)
    joint = run_joint_saes(hidden, base_sae, residual_sae, residual_stats)
    return _aggregate_sample(
        residual_sae, joint.residual_error, roles, stats=residual_stats,
        token_selection=token_selection, batch_size=256, device=device, aggregation=aggregation,
    )


def _top_k_feature_set(vec: torch.Tensor | None, k: int) -> set[int]:
    if vec is None:
        return set()
    nz = vec.nonzero(as_tuple=True)[0]
    if len(nz) == 0:
        return set()
    order = torch.argsort(vec[nz].abs(), descending=True)[:k]
    return {int(nz[i].item()) for i in order.tolist()}


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def run(cfg: EvaluateRLFRConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_visible_device(cfg.gemma.model_device)
    schema = _SCHEMA_BY_TASK[cfg.task_name]

    assert "train" not in cfg.dataset.splits, (
        "Refusing to evaluate on a TRAIN split -- SFT/RLFR were trained on this "
        "exact data (and the Stage 1 probe wrapper's classifier was fit on it "
        "too), so this would be a circular, misleadingly inflated comparison. "
        "Use --dataset.splits test (or validation if no test split exists / is "
        "too small)."
    )
    all_samples = load_meme_samples(cfg.dataset)
    targets = [s for s in all_samples if s.gold_fields.get(s.schema.primary_field.name) is not None]
    if cfg.max_samples is not None:
        targets = targets[: cfg.max_samples]
    assert targets, (
        f"No labeled samples found for {cfg.dataset.datasets}/{cfg.dataset.splits} -- for FHM test, "
        "did you pass --dataset.fhm_labels_path / --dataset.fhm_label_split?"
    )
    gold_labels = [t.gold_fields[t.schema.primary_field.name] for t in targets]
    logger.info("Loaded %d held-out (%s) samples for %s", len(targets), cfg.dataset.splits, cfg.task_name)

    variants: dict[str, list[dict[str, Any]]] = {}
    variant_paths = {"base": "", "sft": cfg.sft_adapter_path, "rlfr": cfg.rlfr_adapter_path}
    for name, adapter_path in variant_paths.items():
        if name != "base" and not adapter_path:
            logger.info("Skipping variant %s (no adapter path given)", name)
            continue
        model, processor = _load_variant(cfg, adapter_path, device)
        rows = _run_classification(model, processor, device, targets, schema)
        variants[name] = rows
        pred_labels = [r["pred"] for r in rows]
        metrics = _per_class_metrics(gold_labels, pred_labels)
        logger.info("Variant %s: macro_f1=%.4f accuracy=%.4f", name, metrics["macro_f1"], metrics["accuracy"])
        (out_dir / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Step 5: compare against the Stage 1 probe wrapper (not just base LVLM) ──
    logger.info("Running Stage 1 probe wrapper for paired comparison")
    monitor = RewardMonitor(cfg.probe, RewardWeights())
    probe_preds = []
    for target in targets:
        assessment = monitor.assess(target.image_path, target.meme_text, schema=schema)
        probe_preds.append(assessment.prediction.label)
    probe_metrics = _per_class_metrics(gold_labels, probe_preds)
    (out_dir / "metrics_probe_wrapper.json").write_text(json.dumps(probe_metrics, indent=2))
    logger.info("Probe wrapper: macro_f1=%.4f", probe_metrics["macro_f1"])

    paired_agreement: dict[str, float] = {}
    for name, rows in variants.items():
        preds = [r["pred"] for r in rows]
        agree = sum(1 for p, q in zip(preds, probe_preds) if p == q) / len(preds)
        paired_agreement[name] = round(agree, 4)
    (out_dir / "paired_agreement_with_probe.json").write_text(json.dumps(paired_agreement, indent=2))

    # ── Identity-group false-positive audit (same convention as validate_reward_monitor.py) ──
    if cfg.identity_group_controls_path:
        controls = _read_jsonl(Path(cfg.identity_group_controls_path))
        controls = [c for c in controls if c["dataset_name"] == cfg.dataset_name]
        if controls:
            fp_rates: dict[str, float] = {}
            for name, adapter_path in variant_paths.items():
                if name != "base" and not adapter_path:
                    continue
                model, processor = _load_variant(cfg, adapter_path, device)
                rows = _run_classification(model, processor, device, controls, schema)
                n_fp = sum(1 for r in rows if r["pred"] == cfg.positive_label)
                fp_rates[name] = round(n_fp / len(rows), 4)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            (out_dir / "identity_group_false_positive_rates.json").write_text(json.dumps(fp_rates, indent=2))
            logger.info("Identity-group control false-positive rates: %s", fp_rates)

    # ── Step 6: reward-hacking check via SAE feature-activation overlap ──
    if cfg.residual_sae_path:
        sae = load_residual_sae(cfg.residual_sae_path, device=device)
        stats = load_residual_stats(cfg.residual_sae_path)
        base_sae = GemmaSae.load_from_hub(cfg.gemma.sae_repo_id, cfg.gemma.sae_hookpoint, device=device)
        sample_subset = targets[: cfg.reward_hacking_n_samples]

        variant_feature_sets: dict[str, list[set[int]]] = {}
        for name, adapter_path in variant_paths.items():
            if name != "base" and not adapter_path:
                continue
            model, processor = _load_variant(cfg, adapter_path, device)
            feats = []
            for sample in sample_subset:
                vec = _extract_feature_vector(
                    model, processor, device, sample, schema, cfg.gemma, base_sae, sae, stats,
                    cfg.reward_hacking_token_selection, cfg.reward_hacking_aggregation,
                )
                feats.append(_top_k_feature_set(vec, cfg.reward_hacking_top_k))
            variant_feature_sets[name] = feats
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        overlap_report: dict[str, float] = {}
        if "base" in variant_feature_sets:
            for name, feats in variant_feature_sets.items():
                if name == "base":
                    continue
                jaccards = [
                    _jaccard(base_set, cand_set)
                    for base_set, cand_set in zip(variant_feature_sets["base"], feats)
                ]
                overlap_report[f"{name}_vs_base_mean_jaccard"] = round(sum(jaccards) / len(jaccards), 4)
        (out_dir / "reward_hacking_feature_overlap.json").write_text(json.dumps(overlap_report, indent=2))
        logger.info("Reward-hacking feature overlap (higher = more consistent with base): %s", overlap_report)

    # ── Step 7: human-audit dump of RLFR false positives/negatives ──
    if "rlfr" in variants:
        model, processor = _load_variant(cfg, cfg.rlfr_adapter_path, device)
        audit_rows = []
        for target, row in zip(targets, variants["rlfr"]):
            gold = target.gold_fields[target.schema.primary_field.name]
            if row["pred"] == gold:
                continue
            m = MemeSample(
                dataset_name=cfg.dataset_name, task_name=schema.task_name, sample_id=target.sample_id,
                split="eval", image_path=target.image_path, meme_text=target.meme_text,
                schema=schema, gold_fields={}, raw_fields={},
            )
            explained = capture_task_hidden_states(model, processor, m, cfg.gemma)
            audit_rows.append({
                "sample_id": target.sample_id,
                "gold": gold,
                "pred": row["pred"],
                "meme_text": target.meme_text,
                "generated_explanation": explained.generated_text,
            })
            if len(audit_rows) >= cfg.human_audit_n_examples:
                break
        with (out_dir / "human_audit_examples.jsonl").open("w") as fd:
            for row in audit_rows:
                fd.write(json.dumps(row) + "\n")
        logger.info("Wrote %d human-audit examples to human_audit_examples.jsonl", len(audit_rows))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Done. Outputs in %s", out_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(EvaluateRLFRConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
