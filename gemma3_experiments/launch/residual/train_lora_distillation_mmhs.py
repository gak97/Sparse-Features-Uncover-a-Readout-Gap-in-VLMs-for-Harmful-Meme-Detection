"""Train a dedicated MMHS LoRA without the universal-adapter trade-offs."""

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.probe_validate import _per_class_metrics
from gemma3_experiments.residual_sae import GemmaResidualConfig, MemeDatasetConfig, MemeSample, build_multimodal_inputs, load_gemma3_model, load_meme_samples, resolve_visible_device
from gemma3_experiments.residual_sae.datasets import MMHS_SCHEMA, build_primary_label_prompt

logger = logging.getLogger(__name__)

TASK_KEY = "MMHS150K::hate_category::label"


@dataclass
class TrainLoraDistillationMMHSConfig:
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    dataset_root: str = "./data"
    targets_dir: str = ""  # optional cached probe targets; empty uses frozen-base soft targets
    output_dir: str = "./lora_distillation_ckpts_mmhs"
    resume_adapter_path: str = ""
    start_step: int = 0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = r".*language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    hard_label_weight: float = 1.0
    soft_label_weight: float = 0.07
    soft_label_temperature: float = 2.0
    subtype_preservation_weight: float = 0.15
    subtype_temperature: float = 2.0
    class_weight_boost: str = "Sexist=1.5,Racist=1.5"
    max_class_weight: float = 8.0
    controls_path: str = "./data/gemma_release_safety_controls/release_safety_controls.jsonl"
    safety_penalty_buckets: str = "direct_threat_or_slur,out_of_domain_harmful"
    lr: float = 1e-4
    max_steps: int = 6000
    grad_accum_steps: int = 8
    grad_clip_norm: float = 1.0
    warmup_steps: int = 200
    eval_every_steps: int = 500
    validation_per_class_cap: int = 300
    seed: int = 42
    device: str = "cuda:0"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fd:
        return [json.loads(line) for line in fd if line.strip()]


def _parse_boosts(spec: str) -> dict[str, float]:
    boosts: dict[str, float] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        label, value = entry.split("=")
        boosts[label.strip()] = float(value)
    return boosts


def _load_safety_controls(path: str, buckets: list[str]) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        logger.warning("Safety controls path %s not found; skipping periodic safety checkpoint check", path)
        return []
    rows = [row for row in _read_jsonl(Path(path)) if row["control"] in buckets]
    logger.info("Loaded %d safety-critical control rows for periodic checkpoint checks: %s", len(rows), Counter(row["control"] for row in rows))
    return rows


def safety_fnr(model: Any, processor: Any, device: str, controls: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    """False-negative rate on held-out safety-critical controls (direct threats/slurs, out-of-domain harmful).

    Mirrors the buckets reported by release_safety_audit.py so a checkpoint can be penalized during training
    for the exact regression that showed up in the 2026-08-20 release audit (adapter learning to say "not hate"
    on genuinely harmful content it was never trained on directly).
    """
    if not controls:
        return {"n": 0, "mean_fnr": 0.0, "by_bucket": {}}
    by_bucket: dict[str, list[bool]] = {}
    for row in controls:
        scores = _score_labels(model, processor, Image.open(row["image_path"]).convert("RGB"), row["meme_text"], labels, device)
        pred_label = labels[int(scores.argmax())]
        false_negative = row["is_harmful"] and pred_label == "NotHate"
        by_bucket.setdefault(row["control"], []).append(false_negative)
    bucket_fnr = {name: sum(flags) / len(flags) for name, flags in by_bucket.items()}
    mean_fnr = sum(bucket_fnr.values()) / len(bucket_fnr) if bucket_fnr else 0.0
    return {"n": len(controls), "mean_fnr": mean_fnr, "by_bucket": bucket_fnr}


def _score_labels(model: Any, processor: Any, image: Image.Image, text: str, labels: list[str], device: str) -> torch.Tensor:
    sample = MemeSample("MMHS150K", "hate_category", "mmhs", "train", "", text, MMHS_SCHEMA, {}, {})
    prompt = build_primary_label_prompt(sample)
    prompt_inputs = build_multimodal_inputs(processor, image, prompt, device)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    scores: list[torch.Tensor] = []
    for label in labels:
        inputs = build_multimodal_inputs(processor, image, prompt, device, assistant_text=label, add_generation_prompt=False)
        label_ids = inputs["input_ids"][0, prompt_len:]
        assert len(label_ids), f"Empty label tokens for {label!r}"
        logits = model(**inputs, use_cache=False).logits[0, prompt_len - 1:prompt_len - 1 + len(label_ids)].float()
        scores.append(torch.log_softmax(logits, dim=-1).gather(1, label_ids.unsqueeze(1)).sum())
    return torch.stack(scores)


def _validation_samples(root: str, cap: int) -> list[Any]:
    samples = [s for s in load_meme_samples(MemeDatasetConfig(root=root, datasets=["MMHS150K"], splits=["validation"])) if s.task_name == "hate_category"]
    selected: list[Any] = []
    n_by_label: Counter[str] = Counter()
    for sample in samples:
        gold = sample.gold_fields.get(sample.schema.primary_field.name)
        if gold is not None and n_by_label[gold] < cap:
            selected.append(sample)
            n_by_label[gold] += 1
    assert selected, "No MMHS validation samples found"
    return selected


def evaluate(model: Any, processor: Any, device: str, samples: list[Any]) -> dict[str, Any]:
    labels = list(MMHS_SCHEMA.primary_field.labels)
    gold: list[str] = []
    pred: list[str] = []
    prob_hate: list[float] = []
    for sample in samples:
        scores = _score_labels(model, processor, Image.open(sample.image_path).convert("RGB"), sample.meme_text, labels, device)
        probs = torch.softmax(scores, dim=0)
        gold.append(sample.gold_fields[sample.schema.primary_field.name])
        pred.append(labels[int(scores.argmax())])
        prob_hate.append(float(1 - probs[labels.index("NotHate")].item()))
    subtype = _per_class_metrics(gold, pred)
    binary_gold = ["Hate" if label != "NotHate" else "NotHate" for label in gold]
    binary_pred = ["Hate" if label != "NotHate" else "NotHate" for label in pred]
    binary = _per_class_metrics(binary_gold, binary_pred)
    selection_score = 2 * binary["macro_f1"] * subtype["macro_f1"] / max(1e-12, binary["macro_f1"] + subtype["macro_f1"])
    return {"binary": binary, "subtype": subtype, "selection_score": selection_score, "n": len(gold), "mean_hate_probability": sum(prob_hate) / len(prob_hate)}


def run(cfg: TrainLoraDistillationMMHSConfig) -> None:
    from peft import LoraConfig, get_peft_model

    if cfg.targets_dir:
        manifest = json.loads((Path(cfg.targets_dir) / "distillation_manifest.json").read_text())
        entries = [entry for entry in manifest if entry["task_key"] == TASK_KEY]
        assert len(entries) == 1, f"Expected one MMHS manifest entry, found {len(entries)}"
        entry = entries[0]
        rows = _read_jsonl(Path(cfg.targets_dir) / f"distillation_targets_{entry['slug']}.jsonl")
        assert rows, "No MMHS targets found"
        labels = list(entry["labels"])
        logger.info("Using cached probe soft targets from %s", cfg.targets_dir)
    else:
        samples = [s for s in load_meme_samples(MemeDatasetConfig(root=cfg.dataset_root, datasets=["MMHS150K"], splits=["train"])) if s.task_name == "hate_category"]
        rows = [{"image_path": s.image_path, "meme_text": s.meme_text, "gold_label": s.gold_fields[s.schema.primary_field.name]} for s in samples]
        labels = list(MMHS_SCHEMA.primary_field.labels)
        logger.info("No targets_dir supplied; using frozen-base temperature distillation on %d raw MMHS rows", len(rows))
    assert labels == list(MMHS_SCHEMA.primary_field.labels), "MMHS target labels do not match the schema"
    counts = Counter(row["gold_label"] for row in rows)
    boosts = _parse_boosts(cfg.class_weight_boost)
    class_weights = torch.tensor(
        [min(cfg.max_class_weight, (len(rows) / (len(labels) * counts[label])) * boosts.get(label, 1.0)) for label in labels],
        device=resolve_visible_device(cfg.device),
    )
    logger.info("Dedicated MMHS proportional sampling: %d rows, class counts=%s, class_weights=%s (capped at %.2f)", len(rows), dict(counts), class_weights.tolist(), cfg.max_class_weight)
    safety_buckets = [b.strip() for b in cfg.safety_penalty_buckets.split(",") if b.strip()]
    safety_controls = _load_safety_controls(cfg.controls_path, safety_buckets)

    device = resolve_visible_device(cfg.device)
    model, processor = load_gemma3_model(cfg.gemma)
    if cfg.resume_adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, cfg.resume_adapter_path, is_trainable=True)
        logger.info("Resuming LoRA weights from %s with a fresh optimizer", cfg.resume_adapter_path)
    else:
        model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=cfg.lora_target_modules, task_type="CAUSAL_LM"))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, cfg.warmup_steps)))
    val_samples = _validation_samples(cfg.dataset_root, cfg.validation_per_class_cap)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    order = list(range(len(rows)))
    rng = random.Random(cfg.seed)
    rng.shuffle(order)
    cursor = 0
    history_path = out_dir / "checkpoint_metrics.json"
    history: list[dict[str, Any]] = json.loads(history_path.read_text()) if history_path.exists() else []
    best_score = max((float(item["selection_score"]) for item in history), default=float("-inf"))
    best_blended_score = max((float(item["blended_score"]) for item in history if "blended_score" in item), default=float("-inf"))

    for step in range(1, cfg.max_steps + 1):
        optimizer.zero_grad()
        mean_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            if cursor == len(order):
                rng.shuffle(order)
                cursor = 0
            row = rows[order[cursor]]
            cursor += 1
            image = Image.open(row["image_path"]).convert("RGB")
            logits = _score_labels(model, processor, image, row["meme_text"], labels, device)
            gold_i = torch.tensor([labels.index(row["gold_label"])], device=device)
            hard_loss = F.cross_entropy(logits.unsqueeze(0), gold_i, weight=class_weights)
            base_logits = None
            if "probe_label_scores" in row:
                teacher = torch.tensor([row["probe_label_scores"][label] for label in labels], device=device).clamp_min(1e-6)
                soft_target = torch.softmax(teacher.log() / cfg.soft_label_temperature, dim=0)
            else:
                with model.disable_adapter(), torch.no_grad():
                    base_logits = _score_labels(model, processor, image, row["meme_text"], labels, device)
                soft_target = F.softmax(base_logits / cfg.soft_label_temperature, dim=0)
            soft_loss = F.kl_div(F.log_softmax(logits / cfg.soft_label_temperature, dim=0), soft_target, reduction="batchmean") * cfg.soft_label_temperature**2
            preservation = torch.zeros((), device=device)
            if row["gold_label"] != "NotHate":
                if base_logits is None:
                    with model.disable_adapter(), torch.no_grad():
                        base_logits = _score_labels(model, processor, image, row["meme_text"], labels, device)
                subtype_i = [i for i, label in enumerate(labels) if label != "NotHate"]
                preservation = F.kl_div(F.log_softmax(logits[subtype_i] / cfg.subtype_temperature, dim=0), F.softmax(base_logits[subtype_i] / cfg.subtype_temperature, dim=0), reduction="batchmean") * cfg.subtype_temperature**2
            loss = cfg.hard_label_weight * hard_loss + cfg.soft_label_weight * soft_loss + cfg.subtype_preservation_weight * preservation
            (loss / cfg.grad_accum_steps).backward()
            mean_loss += float(loss.item())
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        if step % cfg.eval_every_steps:
            continue
        model.eval()
        metrics = evaluate(model, processor, device, val_samples)
        safety = safety_fnr(model, processor, device, safety_controls, labels)
        model.train()
        global_step = cfg.start_step + step
        blended_score = metrics["selection_score"] * (1.0 - safety["mean_fnr"])
        metrics.update({"step": global_step, "train_loss": mean_loss / cfg.grad_accum_steps, "safety_check": safety, "blended_score": blended_score})
        history.append(metrics)
        ckpt = out_dir / f"step_{global_step:06d}"
        model.save_pretrained(str(ckpt))
        if metrics["selection_score"] > best_score:
            best_score = metrics["selection_score"]
            model.save_pretrained(str(out_dir / "best_dual"))
        if blended_score > best_blended_score:
            best_blended_score = blended_score
            model.save_pretrained(str(out_dir / "best_blended"))
        (out_dir / "checkpoint_metrics.json").write_text(json.dumps(history, indent=2))
        logger.info(
            "step=%d loss=%.4f binary_f1=%.4f subtype_f1=%.4f dual=%.4f safety_fnr=%.4f blended=%.4f",
            step, metrics["train_loss"], metrics["binary"]["macro_f1"], metrics["subtype"]["macro_f1"], metrics["selection_score"], safety["mean_fnr"], blended_score,
        )
    model.save_pretrained(str(out_dir / "final"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse(TrainLoraDistillationMMHSConfig, argument_generation_mode=ArgumentGenerationMode.BOTH))


if __name__ == "__main__":
    main()