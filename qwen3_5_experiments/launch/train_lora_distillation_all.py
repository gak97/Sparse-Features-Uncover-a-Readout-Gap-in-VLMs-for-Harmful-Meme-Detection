

import json
import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.generate_lora_distillation_targets_all import (
    TASK_SPECS,
)
from gemma3_experiments.launch.residual.probe_validate import _per_class_metrics
from gemma3_experiments.residual_sae import MemeDatasetConfig, load_meme_samples
from gemma3_experiments.residual_sae.datasets import (
    build_primary_label_prompt,
    build_simple_classify_prompt,
)

from qwen3_5_experiments.config import QwenResidualConfig
from qwen3_5_experiments.qwen import build_multimodal_inputs, load_qwen3_model

logger = logging.getLogger(__name__)

# Qwen3.5-9B-Base architecture facts (confirmed via live inspection,
# 2026-08-25): 8/32 layers use self_attn, the rest use linear_attn.
SELF_ATTN_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)

SCHEMA_KEYS = [
    f"{spec['dataset_name']}::{spec['task_name']}::{spec['field_name']}" for spec in TASK_SPECS
]


def _default_qwen_lora_target_modules() -> str:
    self_attn_alt = "|".join(str(i) for i in SELF_ATTN_LAYERS)
    return (
        r".*language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$"
        r"|.*language_model\.layers\.(" + self_attn_alt + r")\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
    )


@dataclass
class TrainLoraDistillationAllConfig:
    qwen: QwenResidualConfig = field(default_factory=QwenResidualConfig)
    dataset_root: str = "./data"
    output_dir: str = "./lora_distillation_ckpts_all"
    resume_adapter_path: str = ""
    start_step: int = 0

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = field(default_factory=_default_qwen_lora_target_modules)

    hard_label_weight: float = 1.0
    soft_label_weight: float = 0.3
    soft_label_temperature: float = 2.0

    controls_path: str = (
        "./data/gemma_release_safety_controls/release_safety_controls.jsonl"
    )
    safety_penalty_buckets: str = "direct_threat_or_slur,out_of_domain_harmful"

    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 3000
    grad_accum_steps: int = 8
    grad_clip_norm: float = 1.0
    warmup_steps: int = 150
    eval_every_steps: int = 300
    validation_per_class_cap: int = 100
    seed: int = 42
    device: str = "cuda:0"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fd:
        return [json.loads(line) for line in fd if line.strip()]


def _load_safety_controls(path: str, buckets: list[str]) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        logger.warning("Safety controls path %s not found; skipping periodic safety checkpoint check", path)
        return []
    rows = [row for row in _read_jsonl(Path(path)) if row["control"] in buckets]
    logger.info(
        "Loaded %d safety-critical control rows for periodic checkpoint checks: %s",
        len(rows), Counter(row["control"] for row in rows),
    )
    return rows


def _score_all_labels(
    model: Any, processor: Any, image: Image.Image, meme_text: str, spec: dict[str, Any], device: str,
) -> torch.Tensor:
    """Differentiable per-label log-score vector for a single task spec."""
    from gemma3_experiments.residual_sae.datasets import MemeSample

    labels = spec["labels"]
    schema_lookup = _SCHEMA_BY_TASK_KEY[f"{spec['dataset_name']}::{spec['task_name']}::{spec['field_name']}"]
    sample = MemeSample(
        spec["dataset_name"], spec["task_name"], "train", "train", "", meme_text, schema_lookup, {}, {},
    )

    if spec["decode_strategy"] == "yes_no":
        prompt_text = build_simple_classify_prompt(sample)
        inputs = build_multimodal_inputs(processor, image, prompt_text, device)
        outputs = model(**inputs, use_cache=False)
        last_logits = outputs.logits[0, -1].float()
        yes_ids_t, no_ids_t = _YES_NO_IDS
        logit_yes = last_logits.index_select(0, yes_ids_t).max()
        logit_no = last_logits.index_select(0, no_ids_t).max()
        pos_label = spec["yes_no_positive_label"]
        neg_label = next(lbl for lbl in labels if lbl != pos_label)
        scores = {pos_label: logit_yes, neg_label: logit_no}
        return torch.stack([scores[lbl] for lbl in labels])

    prompt_text = build_primary_label_prompt(sample)
    prompt_inputs = build_multimodal_inputs(processor, image, prompt_text, device)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    scores_list: list[torch.Tensor] = []
    for label_text in labels:
        full_inputs = build_multimodal_inputs(
            processor, image, prompt_text, device, assistant_text=label_text, add_generation_prompt=False,
        )
        full_ids = full_inputs["input_ids"][0]
        label_ids = full_ids[prompt_len:]
        assert len(label_ids) > 0, f"Empty label token sequence for {label_text!r}"
        outputs = model(**full_inputs, use_cache=False)
        logits = outputs.logits[0].float()
        log_probs = F.log_softmax(logits[prompt_len - 1: prompt_len - 1 + len(label_ids)], dim=-1)
        token_log_probs = log_probs.gather(1, label_ids.unsqueeze(1)).squeeze(1)
        scores_list.append(token_log_probs.sum())
    return torch.stack(scores_list)


_SCHEMA_BY_TASK_KEY: dict[str, Any] = {}
_YES_NO_IDS: tuple[torch.Tensor, torch.Tensor] | None = None


def _validation_samples(spec: dict[str, Any], root: str, cap: int) -> list[Any]:
    samples = [
        s for s in load_meme_samples(
            MemeDatasetConfig(root=root, datasets=[spec["dataset_name"]], splits=["validation"])
        )
        if s.task_name == spec["task_name"]
    ]
    selected: list[Any] = []
    n_by_label: Counter[str] = Counter()
    for sample in samples:
        gold = sample.gold_fields.get(sample.schema.primary_field.name)
        if gold is not None and n_by_label[gold] < cap:
            selected.append(sample)
            n_by_label[gold] += 1
    return selected


def evaluate_task(model: Any, processor: Any, device: str, spec: dict[str, Any], samples: list[Any]) -> dict[str, Any]:
    labels = list(spec["labels"])
    gold: list[str] = []
    pred: list[str] = []
    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB")
        with torch.no_grad():
            scores = _score_all_labels(model, processor, image, sample.meme_text, spec, device)
        gold.append(sample.gold_fields[sample.schema.primary_field.name])
        pred.append(labels[int(scores.argmax())])
    if not gold:
        return {"n": 0, "macro_f1": 0.0}
    return _per_class_metrics(gold, pred)


def safety_fnr(model: Any, processor: Any, device: str, controls: list[dict[str, Any]]) -> dict[str, Any]:
    """FNR on held-out safety-critical controls, scored with the MMHS150K
    hate_category task spec (the release_safety_controls.jsonl rows carry
    `is_harmful` gold labels that map onto that task's binary collapse).
    Mirrors `train_lora_distillation_mmhs.py::safety_fnr`.
    """
    if not controls:
        return {"n": 0, "mean_fnr": 0.0, "by_bucket": {}}
    mmhs_spec = next(s for s in TASK_SPECS if s["dataset_name"] == "MMHS150K")
    labels = list(mmhs_spec["labels"])
    by_bucket: dict[str, list[bool]] = {}
    for row in controls:
        image = Image.open(row["image_path"]).convert("RGB")
        with torch.no_grad():
            scores = _score_all_labels(model, processor, image, row["meme_text"], mmhs_spec, device)
        pred_label = labels[int(scores.argmax())]
        false_negative = row["is_harmful"] and pred_label == "NotHate"
        by_bucket.setdefault(row["control"], []).append(false_negative)
    bucket_fnr = {name: sum(flags) / len(flags) for name, flags in by_bucket.items()}
    mean_fnr = sum(bucket_fnr.values()) / len(bucket_fnr) if bucket_fnr else 0.0
    return {"n": len(controls), "mean_fnr": mean_fnr, "by_bucket": bucket_fnr}


def run(cfg: TrainLoraDistillationAllConfig) -> None:
    from peft import LoraConfig, get_peft_model

    global _SCHEMA_BY_TASK_KEY, _YES_NO_IDS
    from gemma3_experiments.residual_sae.datasets import (
        CRISIS_A_SCHEMA, CRISIS_B_SCHEMA, HARMEME_SCHEMA, HATEFUL_MEMES_SCHEMA,
        MAMI_SCHEMA, MMHS_SCHEMA, MULTIOFF_SCHEMA,
    )
    _SCHEMA_BY_TASK_KEY = {
        "Hateful_Memes_Dataset::hateful_memes_binary::label": HATEFUL_MEMES_SCHEMA,
        "MAMI::misogyny_and_subtypes::label": MAMI_SCHEMA,
        "MultiOFF::offensive_detection::label": MULTIOFF_SCHEMA,
        "Crisis_HateMM::subtask_a_hate_speech::label": CRISIS_A_SCHEMA,
        "Crisis_HateMM::subtask_b_hate_target::target": CRISIS_B_SCHEMA,
        "HarMeme::harmfulness_and_target::harmfulness": HARMEME_SCHEMA,
        "MMHS150K::hate_category::label": MMHS_SCHEMA,
    }

    device = cfg.qwen.model_device
    model, processor = load_qwen3_model(cfg.qwen)

    from qwen3_5_experiments.qwen import get_yes_no_token_ids
    yes_ids, no_ids = get_yes_no_token_ids(processor)
    _YES_NO_IDS = (torch.tensor(yes_ids, device=device), torch.tensor(no_ids, device=device))

    if cfg.resume_adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.resume_adapter_path, is_trainable=True)
        logger.info("Resuming LoRA weights from %s with a fresh optimizer", cfg.resume_adapter_path)
    else:
        lora_cfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules, task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert trainable_params, "No trainable LoRA parameters found -- check lora_target_modules"
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps))
    )

    logger.info("Loading train samples for %d tasks", len(TASK_SPECS))
    train_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    val_samples_by_task: dict[str, list[Any]] = {}
    for spec in TASK_SPECS:
        task_key = f"{spec['dataset_name']}::{spec['task_name']}::{spec['field_name']}"
        samples = [
            s for s in load_meme_samples(
                MemeDatasetConfig(root=cfg.dataset_root, datasets=[spec["dataset_name"]], splits=["train"])
            )
            if s.task_name == spec["task_name"]
        ]
        rows = [
            {"image_path": s.image_path, "meme_text": s.meme_text, "gold_label": s.gold_fields[s.schema.primary_field.name]}
            for s in samples if s.gold_fields.get(s.schema.primary_field.name) is not None
        ]
        assert rows, f"No train rows for {task_key}"
        train_rows_by_task[task_key] = rows
        val_samples_by_task[task_key] = _validation_samples(spec, cfg.dataset_root, cfg.validation_per_class_cap)
        logger.info("Task %s: %d train rows, %d val samples", task_key, len(rows), len(val_samples_by_task[task_key]))

    safety_buckets = [b.strip() for b in cfg.safety_penalty_buckets.split(",") if b.strip()]
    safety_controls = _load_safety_controls(cfg.controls_path, safety_buckets)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)
    orders: dict[str, list[int]] = {}
    cursors: dict[str, int] = {}
    for task_key, rows in train_rows_by_task.items():
        orders[task_key] = list(range(len(rows)))
        rng.shuffle(orders[task_key])
        cursors[task_key] = 0

    def _next_row(task_key: str) -> dict[str, Any]:
        if cursors[task_key] >= len(orders[task_key]):
            rng.shuffle(orders[task_key])
            cursors[task_key] = 0
        idx = orders[task_key][cursors[task_key]]
        cursors[task_key] += 1
        return train_rows_by_task[task_key][idx]

    history_path = out_dir / "checkpoint_metrics.json"
    history: list[dict[str, Any]] = json.loads(history_path.read_text()) if history_path.exists() else []
    best_score = max((float(item["mean_macro_f1"]) for item in history), default=float("-inf"))
    best_blended_score = max((float(item["blended_score"]) for item in history if "blended_score" in item), default=float("-inf"))

    task_keys = list(train_rows_by_task.keys())
    sample_i = 0
    t_start = time.monotonic()
    accum_loss = 0.0
    step = 0
    optimizer.zero_grad()

    while step < cfg.max_steps:
        task_key = task_keys[sample_i % len(task_keys)]
        spec = next(s for s in TASK_SPECS if f"{s['dataset_name']}::{s['task_name']}::{s['field_name']}" == task_key)
        sample_i += 1
        row = _next_row(task_key)
        labels = spec["labels"]

        image = Image.open(row["image_path"]).convert("RGB")
        logits_k = _score_all_labels(model, processor, image, row["meme_text"], spec, device)
        gold_idx = torch.tensor([labels.index(row["gold_label"])], device=logits_k.device)
        ce_loss = F.cross_entropy(logits_k.unsqueeze(0), gold_idx)

        with model.disable_adapter(), torch.no_grad():
            base_logits = _score_all_labels(model, processor, image, row["meme_text"], spec, device)
        soft_target = F.softmax(base_logits / cfg.soft_label_temperature, dim=0)
        soft_loss = F.kl_div(
            F.log_softmax(logits_k / cfg.soft_label_temperature, dim=0), soft_target, reduction="batchmean",
        ) * cfg.soft_label_temperature ** 2

        loss = cfg.hard_label_weight * ce_loss + cfg.soft_label_weight * soft_loss

        is_accum_boundary = sample_i % cfg.grad_accum_steps == 0
        (loss / cfg.grad_accum_steps).backward()
        accum_loss += float(loss.item())
        del loss, logits_k, base_logits, soft_target

        if is_accum_boundary:
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            step += 1

            if step % 20 == 0:
                mean_loss = accum_loss / (cfg.grad_accum_steps * 20)
                logger.info("step=%d mean_loss=%.4f elapsed=%.1fs", step, mean_loss, time.monotonic() - t_start)
                accum_loss = 0.0

            if step % cfg.eval_every_steps == 0 or step == cfg.max_steps:
                model.eval()
                per_task_f1: dict[str, float] = {}
                for spec2 in TASK_SPECS:
                    tk = f"{spec2['dataset_name']}::{spec2['task_name']}::{spec2['field_name']}"
                    metrics = evaluate_task(model, processor, device, spec2, val_samples_by_task[tk])
                    per_task_f1[tk] = metrics["macro_f1"]
                mean_macro_f1 = sum(per_task_f1.values()) / len(per_task_f1)
                safety = safety_fnr(model, processor, device, safety_controls)
                model.train()

                global_step = cfg.start_step + step
                blended_score = mean_macro_f1 * (1.0 - safety["mean_fnr"])
                record = {
                    "step": global_step, "per_task_macro_f1": per_task_f1, "mean_macro_f1": mean_macro_f1,
                    "safety_check": safety, "blended_score": blended_score,
                }
                history.append(record)
                ckpt = out_dir / f"step_{global_step:06d}"
                model.save_pretrained(str(ckpt))
                if mean_macro_f1 > best_score:
                    best_score = mean_macro_f1
                    model.save_pretrained(str(out_dir / "best_task_f1"))
                if blended_score > best_blended_score:
                    best_blended_score = blended_score
                    model.save_pretrained(str(out_dir / "best_blended"))
                (out_dir / "checkpoint_metrics.json").write_text(json.dumps(history, indent=2))
                logger.info(
                    "step=%d mean_macro_f1=%.4f safety_fnr=%.4f blended=%.4f per_task=%s",
                    global_step, mean_macro_f1, safety["mean_fnr"], blended_score,
                    {k: round(v, 3) for k, v in per_task_f1.items()},
                )

    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    logger.info("Saved final LoRA adapter to %s", final_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(TrainLoraDistillationAllConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
