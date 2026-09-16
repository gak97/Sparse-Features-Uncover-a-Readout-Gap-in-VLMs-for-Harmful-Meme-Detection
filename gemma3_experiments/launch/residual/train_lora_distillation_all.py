

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.residual_sae import (
    GemmaResidualConfig,
    MemeSample,
    build_multimodal_inputs,
    build_primary_label_prompt,
    get_yes_no_token_ids,
    load_gemma3_model,
    resolve_visible_device,
)
from gemma3_experiments.residual_sae.datasets import (
    CRISIS_A_SCHEMA,
    CRISIS_B_SCHEMA,
    HARMEME_SCHEMA,
    HATEFUL_MEMES_SCHEMA,
    MAMI_SCHEMA,
    MMHS_SCHEMA,
    MULTIOFF_SCHEMA,
    build_simple_classify_prompt,
)

logger = logging.getLogger(__name__)

SCHEMA_BY_TASK_KEY: dict[str, Any] = {
    "Hateful_Memes_Dataset::hateful_memes_binary::label": HATEFUL_MEMES_SCHEMA,
    "MAMI::misogyny_and_subtypes::label": MAMI_SCHEMA,
    "MultiOFF::offensive_detection::label": MULTIOFF_SCHEMA,
    "Crisis_HateMM::subtask_a_hate_speech::label": CRISIS_A_SCHEMA,
    "Crisis_HateMM::subtask_b_hate_target::target": CRISIS_B_SCHEMA,
    "HarMeme::harmfulness_and_target::harmfulness": HARMEME_SCHEMA,
    "MMHS150K::hate_category::label": MMHS_SCHEMA,
}


@dataclass
class LoraDistillationAllConfig:
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    targets_dir: str = ""       # output_dir from generate_lora_distillation_targets_all.py
    output_dir: str = "./lora_distillation_ckpts_all"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = (
        r".*language_model\.layers\.\d+\.(self_attn|mlp)\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    )

    hard_label_weight: float = 0.5
    soft_label_weight: float = 0.5
    margin_weight: float = 0.5
    margin: float = 1.0
    # 1 pair/task/step (not 2): margin pairs are now backpropped one at a time
    # (see the training loop's comment) so this mainly controls wall-clock
    # cost, not peak memory, but keeping it conservative leaves more headroom
    # given up to 6 tasks-with-pairs contribute every optimizer step.
    pairs_per_task_per_step: int = 1
    calibration_weight: float = 0.5
    calibration_ema_decay: float = 0.98

    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 4000
    grad_accum_steps: int = 8
    grad_clip_norm: float = 1.0
    warmup_steps: int = 100
    log_every_steps: int = 20
    save_every_steps: int = 500
    seed: int = 42
    device: str = "cuda:0"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fd:
        for line in fd:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_peft():
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "peft is required for LoRA distillation training. Install with: pip install peft"
        ) from exc
    return LoraConfig, get_peft_model


def run(cfg: LoraDistillationAllConfig) -> None:
    LoraConfig, get_peft_model = _load_peft()

    device = resolve_visible_device(cfg.device)
    targets_dir = Path(cfg.targets_dir)
    manifest = json.loads((targets_dir / "distillation_manifest.json").read_text())
    assert manifest, f"Empty distillation_manifest.json in {targets_dir}"

    task_meta_by_key: dict[str, dict[str, Any]] = {}
    targets_by_task: dict[str, list[dict[str, Any]]] = {}
    pairs_by_task: dict[str, list[dict[str, Any]]] = {}
    ema_pos_rate: dict[str, float] = {}

    for entry in manifest:
        task_key = entry["task_key"]
        slug = entry["slug"]
        rows = _read_jsonl(targets_dir / f"distillation_targets_{slug}.jsonl")
        if not rows:
            logger.warning("Task %s has no targets, skipping", task_key)
            continue
        task_meta_by_key[task_key] = entry
        targets_by_task[task_key] = rows
        pairs_by_task[task_key] = _read_jsonl(targets_dir / f"distillation_pairs_{slug}.jsonl")
        if entry.get("calibration") and entry.get("empirical_positive_prior") is not None:
            ema_pos_rate[task_key] = float(entry["empirical_positive_prior"])
        logger.info(
            "Task %s: %d targets, %d pairs, calibration=%s",
            task_key, len(rows), len(pairs_by_task[task_key]), entry.get("calibration"),
        )

    task_keys = list(targets_by_task.keys())
    assert task_keys, "No tasks with targets found -- check targets_dir"
    target_by_sid_per_task = {
        tk: {row["sample_id"]: row for row in rows} for tk, rows in targets_by_task.items()
    }
    tasks_with_pairs = [tk for tk in task_keys if pairs_by_task.get(tk)]
    logger.info("Training on %d tasks; %d have pairwise margin pairs", len(task_keys), len(tasks_with_pairs))

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FROZEN base Gemma 3 LVLM: %s", cfg.gemma.model_id)
    model, processor = load_gemma3_model(cfg.gemma)
    yes_ids, no_ids = get_yes_no_token_ids(processor)
    yes_ids_t = torch.tensor(yes_ids, device=device)
    no_ids_t = torch.tensor(no_ids, device=device)

    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert trainable_params, "No trainable LoRA parameters found -- check lora_target_modules"
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps))
    )

    def _score_all_labels(image: Image.Image, meme_text: str, task_key: str) -> dict[str, torch.Tensor]:
        meta = task_meta_by_key[task_key]
        schema = SCHEMA_BY_TASK_KEY[task_key]
        labels = meta["labels"]
        sample = MemeSample(
            dataset_name=meta["dataset_name"], task_name=meta["task_name"], sample_id="train",
            split="train", image_path="", meme_text=meme_text, schema=schema, gold_fields={}, raw_fields={},
        )
        if meta["decode_strategy"] == "yes_no":
            prompt_text = build_simple_classify_prompt(sample)
            inputs = build_multimodal_inputs(processor, image, prompt_text, device)
            outputs = model(**inputs, use_cache=False)
            last_logits = outputs.logits[0, -1].float()
            logit_yes = last_logits.index_select(0, yes_ids_t).max()
            logit_no = last_logits.index_select(0, no_ids_t).max()
            pos_label = meta["yes_no_positive_label"]
            neg_label = next(lbl for lbl in labels if lbl != pos_label)
            return {pos_label: logit_yes, neg_label: logit_no}

        prompt_text = build_primary_label_prompt(sample)
        prompt_inputs = build_multimodal_inputs(processor, image, prompt_text, device)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        scores: dict[str, torch.Tensor] = {}
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
            scores[label_text] = token_log_probs.sum()
        return scores

    def _positive_class_score(scores: dict[str, torch.Tensor], task_key: str) -> torch.Tensor:
        meta = task_meta_by_key[task_key]
        if meta["decode_strategy"] == "yes_no":
            return scores[meta["semantic_positive_label"]]
        neg_labels = meta.get("collapse_negative_labels") or ()
        pos_labels = [lbl for lbl in meta["labels"] if lbl not in neg_labels]
        return torch.logsumexp(torch.stack([scores[lbl] for lbl in pos_labels]), dim=0)

    rng = random.Random(cfg.seed)
    orders: dict[str, list[int]] = {}
    cursors: dict[str, int] = {}
    for tk in task_keys:
        orders[tk] = list(range(len(targets_by_task[tk])))
        rng.shuffle(orders[tk])
        cursors[tk] = 0

    def _next_row(task_key: str) -> dict[str, Any]:
        if cursors[task_key] >= len(orders[task_key]):
            rng.shuffle(orders[task_key])
            cursors[task_key] = 0
        idx = orders[task_key][cursors[task_key]]
        cursors[task_key] += 1
        return targets_by_task[task_key][idx]

    step = 0
    sample_i = 0
    accum_loss = 0.0
    t_start = time.monotonic()
    optimizer.zero_grad()

    while step < cfg.max_steps:
        task_key = task_keys[sample_i % len(task_keys)]
        sample_i += 1
        row = _next_row(task_key)
        meta = task_meta_by_key[task_key]

        image = Image.open(row["image_path"]).convert("RGB")
        scores = _score_all_labels(image, row["meme_text"], task_key)
        labels = meta["labels"]
        logits_k = torch.stack([scores[lbl] for lbl in labels])

        gold_idx = torch.tensor([labels.index(row["gold_label"])], device=logits_k.device)
        ce_loss = F.cross_entropy(logits_k.unsqueeze(0), gold_idx)

        target_soft = torch.tensor(
            [[row["probe_label_scores"].get(lbl, 0.0) for lbl in labels]], device=logits_k.device,
        )
        log_probs = F.log_softmax(logits_k, dim=-1).unsqueeze(0)
        kl_loss = F.kl_div(log_probs, target_soft, reduction="batchmean")

        loss = cfg.hard_label_weight * ce_loss + cfg.soft_label_weight * kl_loss

        if meta.get("calibration") and task_key in ema_pos_rate:
            probs = F.softmax(logits_k, dim=-1)
            neg_labels = meta.get("collapse_negative_labels") or ()
            pos_prob = sum(probs[labels.index(lbl)] for lbl in labels if lbl not in neg_labels)
            prior = float(task_meta_by_key[task_key]["empirical_positive_prior"])
            overshoot = max(0.0, ema_pos_rate[task_key] - prior)
            if overshoot > 0:
                loss = loss + cfg.calibration_weight * overshoot * pos_prob
            with torch.no_grad():
                ema_pos_rate[task_key] = (
                    cfg.calibration_ema_decay * ema_pos_rate[task_key]
                    + (1.0 - cfg.calibration_ema_decay) * float(pos_prob.item())
                )

        is_accum_boundary = sample_i % cfg.grad_accum_steps == 0

        (loss / cfg.grad_accum_steps).backward()
        accum_loss += float(loss.item())
        del loss, logits_k, log_probs, target_soft, scores

        if tasks_with_pairs and is_accum_boundary and cfg.margin_weight > 0:
            valid_pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
            for pair_task_key in tasks_with_pairs:
                task_pairs = pairs_by_task[pair_task_key]
                by_sid = target_by_sid_per_task[pair_task_key]
                sampled = rng.sample(task_pairs, min(cfg.pairs_per_task_per_step, len(task_pairs)))
                for pair in sampled:
                    pos_row = by_sid.get(pair["positive_sample_id"])
                    neg_row = by_sid.get(pair["negative_sample_id"])
                    if pos_row is None or neg_row is None:
                        continue
                    valid_pairs.append((pair_task_key, pos_row, neg_row))

            n_valid = len(valid_pairs)
            margin_loss_sum = 0.0
            for pair_task_key, pos_row, neg_row in valid_pairs:
                pos_image = Image.open(pos_row["image_path"]).convert("RGB")
                neg_image = Image.open(neg_row["image_path"]).convert("RGB")
                pos_scores = _score_all_labels(pos_image, pos_row["meme_text"], pair_task_key)
                neg_scores = _score_all_labels(neg_image, neg_row["meme_text"], pair_task_key)
                model_margin = (
                    _positive_class_score(pos_scores, pair_task_key)
                    - _positive_class_score(neg_scores, pair_task_key)
                )
                margin_loss = F.relu(cfg.margin - model_margin)
                weight = cfg.margin_weight / (max(1, n_valid) * cfg.grad_accum_steps)
                (weight * margin_loss).backward()
                margin_loss_sum += float(margin_loss.item())
                del pos_scores, neg_scores, model_margin, margin_loss
            if n_valid:
                accum_loss += cfg.margin_weight * (margin_loss_sum / n_valid)

        if is_accum_boundary:
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            step += 1

            if step % cfg.log_every_steps == 0:
                mean_loss = accum_loss / (cfg.grad_accum_steps * cfg.log_every_steps)
                logger.info(
                    "step=%d mean_loss=%.4f elapsed=%.1fs ema_pos_rate=%s",
                    step, mean_loss, time.monotonic() - t_start,
                    {tk: round(v, 3) for tk, v in ema_pos_rate.items()},
                )
                accum_loss = 0.0

            if step % cfg.save_every_steps == 0 or step == cfg.max_steps:
                ckpt_dir = out_dir / f"step_{step:06d}"
                model.save_pretrained(str(ckpt_dir))
                logger.info("Saved LoRA adapter checkpoint to %s", ckpt_dir)

    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    logger.info("Saved final LoRA adapter to %s", final_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(LoraDistillationAllConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
