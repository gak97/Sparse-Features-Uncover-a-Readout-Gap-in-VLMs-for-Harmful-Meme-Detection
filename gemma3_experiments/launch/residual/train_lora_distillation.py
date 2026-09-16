

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


from gemma3_experiments.meme_safety_probe.wrapper import HATEFUL_MEMES_SCHEMA
from gemma3_experiments.residual_sae import (
    GemmaResidualConfig,
    MemeSample,
    build_multimodal_inputs,
    get_yes_no_token_ids,
    load_gemma3_model,
    resolve_visible_device,
)
from gemma3_experiments.residual_sae.datasets import build_simple_classify_prompt

logger = logging.getLogger(__name__)


@dataclass
class LoraDistillationConfig:
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)
    targets_path: str = ""
    pairs_path: str = ""
    output_dir: str = "./lora_distillation_ckpts"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Regex restricted to the LANGUAGE MODEL decoder layers only (not the vision
    # tower, which uses different projection names in Gemma 3's SigLIP encoder
    # e.g. "out_proj"/"fc1"/"fc2" -- this regex will not match those, but the
    # explicit `language_model.layers` prefix is an extra safeguard so LoRA
    # never touches the frozen vision tower).
    lora_target_modules: str = (
        r".*language_model\.layers\.\d+\.(self_attn|mlp)\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    )

    # Loss weighting. hard_label_weight + soft_label_weight need not sum to 1.
    hard_label_weight: float = 0.5   # CE against the gold label
    soft_label_weight: float = 0.5   # KL against the probe's soft P(hateful)
    margin_weight: float = 0.5
    margin: float = 1.0              # target logit gap for the ranking hinge
    pairs_per_margin_step: int = 4    # pairs sampled per optimizer step

    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 2000
    grad_accum_steps: int = 8
    grad_clip_norm: float = 1.0
    warmup_steps: int = 50
    log_every_steps: int = 20
    save_every_steps: int = 250
    seed: int = 42
    device: str = "cuda:0"


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as fd:
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


def run(cfg: LoraDistillationConfig) -> None:
    LoraConfig, get_peft_model = _load_peft()

    device = resolve_visible_device(cfg.device)
    targets = _read_jsonl(cfg.targets_path)
    pairs = _read_jsonl(cfg.pairs_path) if cfg.pairs_path else []
    assert targets, f"No distillation targets found at {cfg.targets_path}"
    logger.info("Loaded %d distillation targets, %d pseudo-confounder pairs", len(targets), len(pairs))
    target_by_sid = {t["sample_id"]: t for t in targets}

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FROZEN base Gemma 3 LVLM: %s", cfg.gemma.model_id)
    model, processor = load_gemma3_model(cfg.gemma)
    yes_ids, no_ids = get_yes_no_token_ids(processor)
    yes_ids_t = torch.tensor(yes_ids, device=device)
    no_ids_t = torch.tensor(no_ids, device=device)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert trainable_params, "No trainable LoRA parameters found -- check lora_target_modules matches real module names"
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps))
    )

    def _forward_logit_yes_no(image_path: str, meme_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        sample = MemeSample(
            dataset_name="external",
            task_name=HATEFUL_MEMES_SCHEMA.task_name,
            sample_id="train",
            split="train",
            image_path=image_path,
            meme_text=meme_text,
            schema=HATEFUL_MEMES_SCHEMA,
            gold_fields={},
            raw_fields={},
        )
        prompt_text = build_simple_classify_prompt(sample)
        image = Image.open(image_path).convert("RGB")
        inputs = build_multimodal_inputs(processor, image, prompt_text, device)
        outputs = model(**inputs, use_cache=False)
        last_logits = outputs.logits[0, -1].float()
        logit_yes = last_logits.index_select(0, yes_ids_t).max()
        logit_no = last_logits.index_select(0, no_ids_t).max()
        return logit_yes, logit_no

    rng = random.Random(cfg.seed)
    order = list(range(len(targets)))
    rng.shuffle(order)

    step = 0
    sample_i = 0
    accum_loss = 0.0
    t_start = time.monotonic()
    optimizer.zero_grad()

    while step < cfg.max_steps:
        if sample_i > 0 and sample_i % len(order) == 0:
            rng.shuffle(order)
        idx = order[sample_i % len(order)]
        sample_i += 1
        target = targets[idx]

        logit_yes, logit_no = _forward_logit_yes_no(target["image_path"], target["meme_text"])
        logits2 = torch.stack([logit_no, logit_yes])

        gold_is_positive = target["gold_label"] == target["positive_label"]
        target_hard = torch.tensor([1 if gold_is_positive else 0], device=logits2.device)
        ce_loss = F.cross_entropy(logits2.unsqueeze(0), target_hard)

        p_hateful = float(target["probe_positive_proba"])
        target_soft = torch.tensor([[1.0 - p_hateful, p_hateful]], device=logits2.device)
        log_probs = F.log_softmax(logits2, dim=-1).unsqueeze(0)
        kl_loss = F.kl_div(log_probs, target_soft, reduction="batchmean")

        loss = cfg.hard_label_weight * ce_loss + cfg.soft_label_weight * kl_loss

        is_accum_boundary = sample_i % cfg.grad_accum_steps == 0
        if pairs and is_accum_boundary and cfg.margin_weight > 0:
            batch_pairs = rng.sample(pairs, min(cfg.pairs_per_margin_step, len(pairs)))
            margin_losses = []
            for pair in batch_pairs:
                h_target = target_by_sid.get(pair["hateful_sample_id"])
                b_target = target_by_sid.get(pair["benign_sample_id"])
                if h_target is None or b_target is None:
                    continue
                h_yes, _ = _forward_logit_yes_no(h_target["image_path"], h_target["meme_text"])
                b_yes, _ = _forward_logit_yes_no(b_target["image_path"], b_target["meme_text"])
                model_margin = h_yes - b_yes
                margin_losses.append(F.relu(cfg.margin - model_margin))
            if margin_losses:
                loss = loss + cfg.margin_weight * torch.stack(margin_losses).mean()

        (loss / cfg.grad_accum_steps).backward()
        accum_loss += float(loss.item())

        if is_accum_boundary:
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % cfg.log_every_steps == 0:
                mean_loss = accum_loss / (cfg.grad_accum_steps * cfg.log_every_steps)
                logger.info(
                    "step=%d mean_loss=%.4f elapsed=%.1fs",
                    step, mean_loss, time.monotonic() - t_start,
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
    cfg = parse(LoraDistillationConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
