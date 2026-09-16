

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


from gemma3_experiments.launch.residual.probe_validate import _read_jsonl
from gemma3_experiments.launch.residual.reward_monitor import RewardMonitor, RewardMonitorConfig, RewardWeights
from gemma3_experiments.meme_safety_probe.wrapper import HATEFUL_MEMES_SCHEMA
from gemma3_experiments.residual_sae import (
    GemmaResidualConfig,
    MemeSample,
    TaskSchema,
    build_multimodal_inputs,
    get_yes_no_token_ids,
    load_gemma3_model,
    resolve_visible_device,
)
from gemma3_experiments.residual_sae.datasets import MAMI_SCHEMA, build_simple_classify_prompt

logger = logging.getLogger(__name__)


@dataclass
class RLFRTaskConfig:
    """One of the pilot's two tasks (FHM or MAMI). Both binary, both already
    have Stage 3 SFT baselines and Stage 1 probes -- extending to a 3rd task
    would need another block like this plus a schema lookup entry."""
    enabled: bool = False
    dataset_name: str = ""
    task_name: str = ""
    targets_path: str = ""    # distillation_targets*.jsonl (reused from Stage 3, no new data gen needed)
    pairs_path: str = ""      # distillation_pairs*.jsonl, "" = no margin loss for this task
    positive_label: str = ""
    negative_label: str = ""

    # Per-task reward-monitor CLASSIFIER profile (fixed 2026-07-29 -- FHM and
    # MAMI need DIFFERENT sklearn classifiers/label spaces even though they
    # share the same frozen LVLM + residual SAE checkpoint; see
    # `MemeSafetyProbe.swap_classifier`'s docstring for the bug this fixes).
    probe_dir: str = ""
    probe_filename: str = "probe.joblib"
    scaler_filename: str = "scaler.joblib"
    label_encoder_filename: str = "label_encoder.joblib"
    token_selection: str = "all"
    aggregation: str = "max"
    calibrated_threshold: float | None = None


@dataclass
class RLFRConfig:
    gemma: GemmaResidualConfig = field(default_factory=GemmaResidualConfig)   # trainable policy
    reward_monitor: RewardMonitorConfig = field(default_factory=RewardMonitorConfig)  # frozen, separate device
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    fhm: RLFRTaskConfig = field(default_factory=lambda: RLFRTaskConfig(
        dataset_name="Hateful_Memes_Dataset", task_name="hateful_memes_binary",
        positive_label="hateful", negative_label="not hateful",
    ))
    mami: RLFRTaskConfig = field(default_factory=lambda: RLFRTaskConfig(
        dataset_name="MAMI", task_name="misogyny_and_subtypes",
        positive_label="misogynous", negative_label="non-misogynous",
    ))

    output_dir: str = "./rlfr_ckpts"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = (
        r".*language_model\.layers\.\d+\.(self_attn|mlp)\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    )

    group_size: int = 4          # GRPO group size (K stochastic completions per prompt)
    temperature: float = 1.0
    kl_coef: float = 0.05        # KL-to-reference regularizer, anti reward-hacking
    entropy_coef: float = 0.02   # exploration floor -- see module docstring "2026-07-29 fixes"

    sft_adapter_path: str = ""

    # Hard gate (2026-07-29): paths to reward_monitor_validation.json files
    # from validate_reward_monitor.py. Training refuses to start if ANY of
    # them report gate_passed=false. Previously this was only a documented
    # recommendation -- FHM's 63.7% identity-group FP rate should have
    # blocked the 2026-07-25 run but did not, since nothing enforced it.
    require_gate_passed_paths: list[str] = field(default_factory=list)

    margin_weight: float = 0.5
    margin: float = 1.0
    pairs_per_margin_step: int = 4

    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 1000
    grad_accum_steps: int = 8
    grad_clip_norm: float = 1.0
    warmup_steps: int = 50
    log_every_steps: int = 20
    save_every_steps: int = 200
    seed: int = 42
    device: str = "cuda:0"       # trainable policy device -- keep separate from reward_monitor.probe.device


def _load_peft():
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required for RLFR training. Install with: pip install peft") from exc
    return LoraConfig, get_peft_model


_SCHEMA_BY_TASK: dict[str, TaskSchema] = {
    "hateful_memes_binary": HATEFUL_MEMES_SCHEMA,
    "misogyny_and_subtypes": MAMI_SCHEMA,
}


def _pair_member_ids(pair: dict[str, Any]) -> tuple[str, str]:
    if "hateful_sample_id" in pair:
        return pair["hateful_sample_id"], pair["benign_sample_id"]
    return pair["positive_sample_id"], pair["negative_sample_id"]


def _make_pool(task_cfg: RLFRTaskConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], TaskSchema]:
    targets = _read_jsonl(Path(task_cfg.targets_path))
    assert targets, f"No targets found at {task_cfg.targets_path}"
    pairs = _read_jsonl(Path(task_cfg.pairs_path)) if task_cfg.pairs_path else []
    schema = _SCHEMA_BY_TASK[task_cfg.task_name]
    logger.info("Task %s::%s: %d targets, %d pairs", task_cfg.dataset_name, task_cfg.task_name, len(targets), len(pairs))
    return targets, pairs, schema


def _check_reward_monitor_gates(paths: list[str]) -> None:
    """Hard gate (2026-07-29): refuse to start training if any referenced
    validate_reward_monitor.py output reports gate_passed=false."""
    for path in paths:
        data = json.loads(Path(path).read_text())
        gate_passed = data.get("gate_passed")
        assert gate_passed is not None, (
            f"{path} has no 'gate_passed' field -- re-run validate_reward_monitor.py "
            "with --identity_group_controls_path set (gate_passed is only computed "
            "when the identity-group check runs)."
        )
        assert gate_passed, (
            f"REWARD MONITOR GATE FAILED for {path}: identity_group_false_positive_rate="
            f"{data.get('identity_group_false_positive_rate')} exceeds threshold "
            f"{data.get('identity_group_fp_threshold')}. Refusing to start RLFR training "
            "with an unsafe reward monitor."
        )
        logger.info("Reward-monitor gate PASSED: %s (identity-group FP rate=%.4f)",
                    path, data.get("identity_group_false_positive_rate", -1.0))


def run(cfg: RLFRConfig) -> None:
    LoraConfig, get_peft_model = _load_peft()
    device = resolve_visible_device(cfg.device)
    rng = random.Random(cfg.seed)

    if cfg.require_gate_passed_paths:
        _check_reward_monitor_gates(cfg.require_gate_passed_paths)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_pools: list[tuple[RLFRTaskConfig, list[dict[str, Any]], list[dict[str, Any]], TaskSchema]] = []
    for task_cfg in (cfg.fhm, cfg.mami):
        if not task_cfg.enabled:
            continue
        targets, pairs, schema = _make_pool(task_cfg)
        task_pools.append((task_cfg, targets, pairs, schema))
    assert task_pools, "No tasks enabled -- set --fhm.enabled true and/or --mami.enabled true"

    logger.info("Loading FROZEN reward monitor (separate device=%s)", cfg.reward_monitor.probe.device)
    monitor = RewardMonitor(cfg.reward_monitor, cfg.reward_weights)

    logger.info("Loading TRAINABLE base Gemma 3 policy: %s (device=%s)", cfg.gemma.model_id, device)
    model, processor = load_gemma3_model(cfg.gemma)
    yes_ids, no_ids = get_yes_no_token_ids(processor)
    yes_ids_t = torch.tensor(yes_ids, device=device)
    no_ids_t = torch.tensor(no_ids, device=device)

    if cfg.sft_adapter_path:
        from peft import PeftModel
        logger.info("Merging Stage 3 SFT adapter as the reference/starting point: %s", cfg.sft_adapter_path)
        model = PeftModel.from_pretrained(model, cfg.sft_adapter_path)
        model = model.merge_and_unload()
    else:
        logger.warning(
            "No --sft_adapter_path given -- KL-to-reference will use the raw "
            "no-LoRA base, which was confirmed badly miscalibrated for MAMI "
            "(91.4%% positive-rate bias) and contributed to the 2026-07-25 "
            "mode collapse. Strongly recommend setting --sft_adapter_path."
        )

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

    def _two_logits(inputs: dict[str, Any]) -> torch.Tensor:
        outputs = model(**inputs, use_cache=False)
        last_logits = outputs.logits[0, -1].float()
        logit_yes = last_logits.index_select(0, yes_ids_t).max()
        logit_no = last_logits.index_select(0, no_ids_t).max()
        return torch.stack([logit_no, logit_yes])  # index 0=no(negative), 1=yes(positive)

    def _build_prompt_inputs(task_cfg: RLFRTaskConfig, schema: TaskSchema, image_path: str, meme_text: str) -> dict[str, Any]:
        sample = MemeSample(
            dataset_name=task_cfg.dataset_name, task_name=task_cfg.task_name,
            sample_id="rlfr_train", split="train", image_path=image_path, meme_text=meme_text,
            schema=schema, gold_fields={}, raw_fields={},
        )
        prompt_text = build_simple_classify_prompt(sample)
        image = Image.open(image_path).convert("RGB")
        return build_multimodal_inputs(processor, image, prompt_text, device)

    step = 0
    sample_i = 0
    accum_reward = 0.0
    accum_loss = 0.0
    t_start = time.monotonic()
    optimizer.zero_grad()

    while step < cfg.max_steps:
        task_cfg, targets, pairs, schema = task_pools[sample_i % len(task_pools)]
        target = targets[rng.randrange(len(targets))]
        sample_i += 1

        if monitor.probe is not None:
            monitor.probe.swap_classifier(
                probe_dir=task_cfg.probe_dir,
                probe_filename=task_cfg.probe_filename,
                scaler_filename=task_cfg.scaler_filename,
                label_encoder_filename=task_cfg.label_encoder_filename,
                token_selection=task_cfg.token_selection,
                aggregation=task_cfg.aggregation,
                positive_label=task_cfg.positive_label,
                calibrated_threshold=task_cfg.calibrated_threshold,
            )
        assessment = monitor.assess(target["image_path"], target["meme_text"], schema=schema)
        task_key = f"{task_cfg.dataset_name}::{task_cfg.task_name}::label"

        inputs = _build_prompt_inputs(task_cfg, schema, target["image_path"], target["meme_text"])

        with torch.no_grad(), model.disable_adapter():
            ref_logits2 = _two_logits(inputs) / cfg.temperature
            ref_dist = torch.distributions.Categorical(logits=ref_logits2)

        completions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
        for _ in range(cfg.group_size):
            logits2 = _two_logits(inputs) / cfg.temperature
            dist = torch.distributions.Categorical(logits=logits2)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            kl = torch.distributions.kl_divergence(dist, ref_dist)
            student_is_positive = bool(action.item() == 1)

            reward = monitor.compute_reward(
                task_key=task_key,
                assessment=assessment,
                gold_label=target["gold_label"],
                positive_label=task_cfg.positive_label,
                negative_label=task_cfg.negative_label,
                student_is_positive=student_is_positive,
                generated_text=assessment.prediction.generated_text,
            )
            completions.append((log_prob, kl, entropy, reward.total))

        rewards = [c[3] for c in completions]
        baseline = sum(rewards) / len(rewards)
        accum_reward += baseline

        is_accum_boundary = sample_i % cfg.grad_accum_steps == 0
        for log_prob, kl, entropy, reward_total in completions:
            advantage = reward_total - baseline
            # Entropy bonus (2026-07-29 fix): without this, the group-relative
            # advantage mechanic self-reinforces toward a fully deterministic
            # A collapsed policy can become deterministic within roughly 1000 steps.
            # "2026-07-29" entry. Subtracting entropy_coef * entropy keeps a
            # floor on exploration.
            loss = -log_prob * advantage + cfg.kl_coef * kl - cfg.entropy_coef * entropy
            (loss / (cfg.group_size * cfg.grad_accum_steps)).backward()
            accum_loss += float(loss.item())

        if pairs and is_accum_boundary and cfg.margin_weight > 0:
            batch_pairs = rng.sample(pairs, min(cfg.pairs_per_margin_step, len(pairs)))
            target_by_sid = {t["sample_id"]: t for t in targets}
            margin_losses = []
            for pair in batch_pairs:
                h_sid, b_sid = _pair_member_ids(pair)
                h_target = target_by_sid.get(h_sid)
                b_target = target_by_sid.get(b_sid)
                if h_target is None or b_target is None:
                    continue
                h_inputs = _build_prompt_inputs(task_cfg, schema, h_target["image_path"], h_target["meme_text"])
                b_inputs = _build_prompt_inputs(task_cfg, schema, b_target["image_path"], b_target["meme_text"])
                h_yes = _two_logits(h_inputs)[1]
                b_yes = _two_logits(b_inputs)[1]
                margin_loss = F.relu(cfg.margin - (h_yes - b_yes))
                weight = cfg.margin_weight / max(1, len(batch_pairs))
                (weight * margin_loss / cfg.grad_accum_steps).backward()
                margin_losses.append(float(margin_loss.item()))

        if is_accum_boundary:
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % cfg.log_every_steps == 0:
                mean_reward = accum_reward / (cfg.grad_accum_steps * cfg.log_every_steps)
                mean_loss = accum_loss / (cfg.group_size * cfg.grad_accum_steps * cfg.log_every_steps)
                logger.info(
                    "step=%d mean_group_reward=%.4f mean_loss=%.4f elapsed=%.1fs",
                    step, mean_reward, mean_loss, time.monotonic() - t_start,
                )
                accum_reward = 0.0
                accum_loss = 0.0

            if step % cfg.save_every_steps == 0 or step == cfg.max_steps:
                ckpt_dir = out_dir / f"step_{step:06d}"
                model.save_pretrained(str(ckpt_dir))
                logger.info("Saved LoRA adapter checkpoint to %s", ckpt_dir)

    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    logger.info("Saved final RLFR LoRA adapter to %s", final_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(RLFRConfig, argument_generation_mode=ArgumentGenerationMode.BOTH)
    run(cfg)


if __name__ == "__main__":
    main()
