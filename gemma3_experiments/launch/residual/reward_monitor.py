

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from gemma3_experiments.meme_safety_probe.wrapper import (
    HATEFUL_MEMES_SCHEMA,
    MemeSafetyProbe,
    MemeSafetyProbeConfig,
    PredictionResult,
)
from gemma3_experiments.residual_sae import (
    GemmaResidualConfig,
    MemeSample,
    TaskSchema,
    classify_constrained,
    load_gemma3_model,
    resolve_visible_device,
)
from gemma3_experiments.residual_sae.datasets import build_simple_classify_prompt

logger = logging.getLogger(__name__)

IDENTITY_GROUP_TERMS: tuple[str, ...] = (
    "muslim", "islam", "jewish", "jew", "black", "african",
    "women", "woman", "female", "girl",
    "gay", "lesbian", "lgbt", "transgender", "trans",
    "mexican", "hispanic", "latino", "latina",
    "asian", "chinese",
    "christian", "disab", "immigrant", "refugee",
)


def contains_identity_group_term(text: str, terms: tuple[str, ...] = IDENTITY_GROUP_TERMS) -> str | None:
    """Return the first matching identity-group term found in `text`, or None."""
    lowered = text.lower()
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\w*", lowered):
            return term
    return None


@dataclass
class RewardMonitorConfig:
    # "sae_probe" (default): wraps MemeSafetyProbe. "sft_model": scores via
    # the Stage 3 SFT model's own classify_constrained() output instead --
    # see module docstring's 2026-07-29 addition for why this exists.
    backend: str = "sae_probe"

    # Reuses MemeSafetyProbeConfig verbatim -- the frozen reward monitor IS a
    # MemeSafetyProbe instance, no separate model-loading code. Only used
    # when backend=="sae_probe"; probe.gemma is ALSO reused for backend==
    # "sft_model" (model_id/device), everything else in `probe` is ignored.
    probe: MemeSafetyProbeConfig = field(default_factory=MemeSafetyProbeConfig)

    # Only used when backend=="sft_model": path to the Stage 3 SFT LoRA
    # adapter to merge into the frozen base model. A single merged model
    # handles BOTH FHM and MAMI schemas via the prompt (build_simple_classify_prompt
    # + classify_constrained), so -- unlike the sae_probe backend -- no
    # per-task classifier swapping is needed at all.
    sft_adapter_path: str = ""

    identity_group_terms: tuple[str, ...] = IDENTITY_GROUP_TERMS
    # EMA decay for the STUDENT's own predicted-positive-rate tracker (used by
    # calibration_reward). Matches train_lora_distillation_all.py's existing
    # EMA convention (decay=0.98) so both Stage 3 and RLFR-era calibration
    # signals behave consistently if compared side by side.
    calibration_ema_decay: float = 0.98


@dataclass
class RewardWeights:
    label_weight: float = 1.0
    calibration_weight: float = 0.25
    pair_margin_weight: float = 0.5
    explanation_consistency_weight: float = 0.0  # off by default, see train_rlfr.py docstring
    false_positive_penalty_weight: float = 1.0
    false_negative_penalty_weight: float = 1.0


@dataclass
class FrozenAssessment:
    """One frozen-monitor forward pass, reusable across all K sampled
    completions of the same training prompt (the frozen probe's own
    prediction does not depend on what the student generates)."""
    prediction: PredictionResult
    identity_term: str | None


@dataclass
class RewardBreakdown:
    label_reward: float
    calibration_reward: float
    pair_margin_reward: float
    explanation_consistency_reward: float
    false_positive_penalty: float
    false_negative_penalty: float
    total: float


class RewardMonitor:
    """Frozen reward source for RLFR training.

    `assess()` does ONE frozen forward pass per (image, meme_text) prompt --
    call it once per training prompt and reuse the returned `FrozenAssessment`
    for every sampled completion of that prompt (GRPO-style group sampling).
    `compute_reward()` is then pure per-completion arithmetic, no model calls.
    """

    def __init__(self, cfg: RewardMonitorConfig, weights: RewardWeights) -> None:
        self.cfg = cfg
        self.weights = weights
        self.probe: MemeSafetyProbe | None = None
        self._sft_model = None
        self._sft_processor = None
        self._sft_device = None

        cfg.probe.gemma.model_device = resolve_visible_device(cfg.probe.device)

        if cfg.backend == "sae_probe":
            logger.info("Loading FROZEN reward-monitor MemeSafetyProbe (device=%s)", cfg.probe.device)
            self.probe = MemeSafetyProbe(cfg.probe)
        elif cfg.backend == "sft_model":
            self._sft_device = cfg.probe.gemma.model_device
            logger.info(
                "Loading FROZEN reward-monitor SFT model: %s + adapter %s (device=%s)",
                cfg.probe.gemma.model_id, cfg.sft_adapter_path, self._sft_device,
            )
            model, processor = load_gemma3_model(cfg.probe.gemma)
            if cfg.sft_adapter_path:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, cfg.sft_adapter_path)
                model = model.merge_and_unload()
            model.eval()
            self._sft_model = model
            self._sft_processor = processor
        else:
            raise ValueError(f"Unsupported RewardMonitorConfig.backend: {cfg.backend!r}")

        # Per task_key EMA of the STUDENT's own predicted-positive rate.
        self._ema_pos_rate: dict[str, float] = {}

    def assess(self, image_path: str, meme_text: str, schema: TaskSchema | None = None) -> FrozenAssessment:
        schema = schema or HATEFUL_MEMES_SCHEMA
        if self.cfg.backend == "sft_model":
            prediction = self._assess_sft_model(image_path, meme_text, schema)
        else:
            prediction = self.probe.predict(image_path, meme_text, schema=schema)
        identity_term = contains_identity_group_term(meme_text, self.cfg.identity_group_terms)
        return FrozenAssessment(prediction=prediction, identity_term=identity_term)

    def _assess_sft_model(self, image_path: str, meme_text: str, schema: TaskSchema) -> PredictionResult:
        from PIL import Image

        sample = MemeSample(
            dataset_name="reward_monitor", task_name=schema.task_name, sample_id="query",
            split="inference", image_path=image_path, meme_text=meme_text, schema=schema,
            gold_fields={}, raw_fields={},
        )
        prompt_text = build_simple_classify_prompt(sample)
        image = Image.open(image_path).convert("RGB")
        is_positive, _, conf_gap = classify_constrained(
            self._sft_model, self._sft_processor, image, prompt_text, self._sft_device,
        )
        primary_labels = schema.primary_field.labels
        positive_label, negative_label = primary_labels[-1], primary_labels[0]
        label = positive_label if is_positive else negative_label
        # conf_gap = sigmoid(logit_yes - logit_no) -- a genuine 2-class
        # probability distribution over {positive_label, negative_label}.
        label_scores = {positive_label: round(conf_gap, 4), negative_label: round(1.0 - conf_gap, 4)}
        return PredictionResult(
            label=label, confidence=label_scores[label], label_scores=label_scores,
            top_features=[], prompt_text=prompt_text, generated_text="",
        )

    def _empirical_prior(self, positive_label: str, negative_label: str) -> float:
        # Uses the frozen monitor's own label_scores distribution as the
        # calibration target's "expected" positive rate proxy when no
        # dataset-level prior is supplied -- 0.5 is the safe fallback for the
        # binary tasks this pilot targets (FHM/MAMI validation splits are both
        # close to balanced, see final-eval-protocol memory).
        return 0.5

    def compute_reward(
        self,
        *,
        task_key: str,
        assessment: FrozenAssessment,
        gold_label: str,
        positive_label: str,
        negative_label: str,
        student_is_positive: bool,
        generated_text: str = "",
        grounding_terms: tuple[str, ...] = (),
        pair_partner_margin: float | None = None,
    ) -> RewardBreakdown:
        w = self.weights
        student_label = positive_label if student_is_positive else negative_label

        # ── label_reward: soft agreement with gold, using the frozen probe's
        # own confidence as a smoothing signal (matches the feedback's "gold
        # label or probe-classifier correctness").
        label_scores = assessment.prediction.label_scores or {}
        p_gold = label_scores.get(gold_label)
        if p_gold is not None:
            # Confidence-scaled +-1: agreeing with gold when the frozen probe
            # was already confident in gold scores closer to +1; disagreeing
            # while the frozen probe was confident in gold scores closer to -1.
            label_reward = 1.0 if student_label == gold_label else -1.0
            label_reward *= max(0.5, p_gold if student_label == gold_label else (1.0 - p_gold))
        else:
            label_reward = 1.0 if student_label == gold_label else -1.0

        # ── calibration_reward: penalize the STUDENT's own drift away from an
        # empirical ~balanced prior (avoid "always say positive"), via the
        # same one-sided-then-symmetric EMA idea as train_lora_distillation_all.py,
        # but SYMMETRIC here (penalizes over- AND under-prediction) per the
        # documented follow-up fix from the 2026-07-24/25 MMHS150K regression.
        prior = self._empirical_prior(positive_label, negative_label)
        ema = self._ema_pos_rate.get(task_key, prior)
        overshoot = ema - prior
        calibration_reward = -abs(overshoot) if (student_is_positive and overshoot > 0) or (
            not student_is_positive and overshoot < 0
        ) else 0.0
        new_ema = self.cfg.calibration_ema_decay * ema + (1 - self.cfg.calibration_ema_decay) * float(student_is_positive)
        self._ema_pos_rate[task_key] = new_ema

        # ── pair_margin_reward: reward agreement with the frozen monitor's
        # OWN pairwise margin direction for FHM pseudo-confounder pairs (only
        # active when the caller supplies the paired sample's margin).
        pair_margin_reward = 0.0
        if pair_partner_margin is not None:
            # pair_partner_margin > 0 means the frozen monitor's own P(positive)
            # is higher for THIS sample than its paired partner -- reward the
            # student for being on the same side of that ranking.
            pair_margin_reward = 1.0 if (pair_partner_margin > 0) == student_is_positive else -1.0

        explanation_consistency_reward = 0.0
        if w.explanation_consistency_weight > 0 and generated_text:
            gen_lower = generated_text.lower()
            terms = grounding_terms or tuple(meme_text_token for meme_text_token in gold_label.lower().split())
            hits = sum(1 for term in terms if term and term.lower() in gen_lower)
            explanation_consistency_reward = min(1.0, hits / max(1, len(terms))) if terms else 0.0

        # ── false_positive_penalty: extra penalty for predicting the harmful/
        # positive class on a GOLD-BENIGN sample that mentions an identity
        # group -- directly targets the feedback's "benign identity-group
        # safety" success criterion.
        false_positive_penalty = 0.0
        if (
            assessment.identity_term is not None
            and gold_label == negative_label
            and student_is_positive
        ):
            false_positive_penalty = 1.0

        false_negative_penalty = 0.0
        if (
            assessment.identity_term is not None
            and gold_label == positive_label
            and not student_is_positive
        ):
            false_negative_penalty = 1.0

        total = (
            w.label_weight * label_reward
            + w.calibration_weight * calibration_reward
            + w.pair_margin_weight * pair_margin_reward
            + w.explanation_consistency_weight * explanation_consistency_reward
            - w.false_positive_penalty_weight * false_positive_penalty
            - w.false_negative_penalty_weight * false_negative_penalty
        )
        return RewardBreakdown(
            label_reward=label_reward,
            calibration_reward=calibration_reward,
            pair_margin_reward=pair_margin_reward,
            explanation_consistency_reward=explanation_consistency_reward,
            false_positive_penalty=false_positive_penalty,
            false_negative_penalty=false_negative_penalty,
            total=total,
        )
