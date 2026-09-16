import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen3_5ForConditionalGeneration,
)

from .config import QwenResidualConfig
from gemma3_experiments.residual_sae.datasets import MemeSample


BASE_IMG_TOKENS = 256  # Approximate image token count for Qwen3.5 vision encoder


@dataclass
class QwenTaskRun:
    hidden: torch.Tensor
    output_ids: torch.Tensor
    input_len: int
    image_span: tuple[int, int]
    generated_text: str
    prompt_text: str


@dataclass
class QwenSaeConfig:
    d_model: int = 4096
    d_sae: int = 65536
    k: int = 100
    architecture: str = "topk"


class QwenSae(nn.Module):
    pass

    def __init__(self, cfg: QwenSaeConfig, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(torch.zeros(cfg.d_sae, cfg.d_model, device=device))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae, device=device))
        self.W_dec = nn.Parameter(torch.zeros(cfg.d_model, cfg.d_sae, device=device))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model, device=device))
        self._k = cfg.k

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """(..., d_model) → (..., d_sae) pre-activations."""
        sae_in = x.to(self.W_enc.dtype) - self.b_dec
        return sae_in @ self.W_enc.T + self.b_enc

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """TopK encode: keep top-k pre-activations, zero out the rest."""
        pre = self.pre_acts(x)
        top_values, top_indices = torch.topk(pre, k=self._k, dim=-1, sorted=False)
        acts = torch.zeros_like(pre)
        acts.scatter_(dim=-1, index=top_indices, src=top_values)
        return acts, top_indices

    def decode(self, acts: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        """(..., d_sae) → (..., d_model) reconstruction."""
        return acts.to(self.W_dec.dtype) @ self.W_dec.T + self.b_dec

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts, indices = self.encode(x)
        return acts, self.decode(acts, indices)

    @staticmethod
    def load_from_hub(repo_id: str, layer: int, device: str = "cpu") -> "QwenSae":
        """Load a Qwen SAE checkpoint from HuggingFace Hub.

        Downloaded from repo_id, filename = layer{n}.sae.pt.
        """
        filename = f"layer{layer}.sae.pt"
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        state_dict = torch.load(local_path, map_location="cpu", weights_only=True)
        cfg = QwenSaeConfig(
            d_model=state_dict["W_dec"].shape[0],
            d_sae=state_dict["W_enc"].shape[0],
        )
        sae = QwenSae(cfg, device=device)
        sae.W_enc.data.copy_(state_dict["W_enc"].to(device))
        sae.b_enc.data.copy_(state_dict["b_enc"].to(device))
        sae.W_dec.data.copy_(state_dict["W_dec"].to(device))
        sae.b_dec.data.copy_(state_dict["b_dec"].to(device))
        sae.eval()
        for param in sae.parameters():
            param.requires_grad_(False)
        return sae


def load_qwen3_model(
    cfg: QwenResidualConfig,
) -> tuple[Qwen3_5ForConditionalGeneration, Any]:
    """Load the Qwen3.5-9B-Base multimodal model and its processor."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map={"": cfg.model_device},
        trust_remote_code=True,
    ).eval()
    return model, processor


def resolve_target_module(
    model: Qwen3_5ForConditionalGeneration, cfg: QwenResidualConfig
) -> nn.Module:
    """Resolve the target transformer layer submodule.

    Qwen3.5-9B-Base structure: model → model (inner) → layers[N]
    """
    candidates = [
        cfg.hook_module_path,
        f"model.language_model.layers.{cfg.sae_layer}",
        f"model.language_model.model.layers.{cfg.sae_layer}",
        f"model.model.layers.{cfg.sae_layer}",
        f"model.layers.{cfg.sae_layer}",
        "model.language_model.layers.20",
    ]
    for path in candidates:
        try:
            return model.get_submodule(path)
        except AttributeError:
            continue
    raise RuntimeError(
        f"Could not find a Qwen residual stream module. Tried: {candidates}"
    )


def _image_token_span(
    input_ids: torch.Tensor, processor: Any, seq_len: int
) -> tuple[int, int]:
    """Detect the image token span in the input sequence.

    Qwen3.5 uses vision_start_token_id / vision_end_token_id to delimit
    image tokens, similar to Gemma 3.
    """
    image_seq_len = getattr(processor, "image_seq_length", BASE_IMG_TOKENS)

    # Try vision_start_token_id / vision_end_token_id
    start_tok = getattr(processor, "vision_start_token_id", None)
    end_tok = getattr(processor, "vision_end_token_id", None)
    if start_tok is not None and end_tok is not None:
        start_positions = (input_ids == start_tok).nonzero(as_tuple=False)
        end_positions = (input_ids == end_tok).nonzero(as_tuple=False)
        if start_positions.numel() > 0 and end_positions.numel() > 0:
            start = int(start_positions[0].item())
            end = int(end_positions[-1].item()) + 1
            return start, min(end, seq_len)

    # Fallback: image_seq_len heuristic
    image_tok_id = getattr(processor, "image_token_id", None)
    if image_tok_id is not None:
        if isinstance(image_tok_id, str):
            image_tok_id = processor.tokenizer.convert_tokens_to_ids(image_tok_id)
        if isinstance(image_tok_id, int):
            positions = (input_ids == image_tok_id).nonzero(as_tuple=False)
            if positions.numel() > 0:
                start = int(positions[0].item())
                return start, min(start + image_seq_len, seq_len)

    # Last resort: find the longest repeated-token run (image placeholder pattern)
    ids = input_ids.tolist()
    best_start, best_len, best_tok = 0, 0, -1
    i = 0
    while i < len(ids):
        j = i
        while j < len(ids) and ids[j] == ids[i]:
            j += 1
        if (j - i) > best_len:
            best_len = j - i
            best_start = i
            best_tok = ids[i]
        i = j
    if best_len >= image_seq_len // 2 and best_tok > 3:
        return best_start, min(best_start + best_len, seq_len)
    return 4, min(4 + BASE_IMG_TOKENS, seq_len)


def build_multimodal_inputs(
    processor: Any,
    image: Any,
    prompt_text: str,
    device: str,
    *,
    assistant_text: str = "",
    add_generation_prompt: bool = True,
) -> dict[str, Any]:
    """Build multimodal inputs (image + text) for Qwen3.5-9B-Base.

    Follows the same pattern as Gemma 3's build_multimodal_inputs.
    The user message contains both the image and the text prompt.
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    if assistant_text:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        )
    # Qwen3VLProcessor does not expose chat_template at the processor level
    # (only processor.tokenizer has it), so processor.apply_chat_template()
    # raises ValueError. Use the two-step pattern instead: format via the
    # tokenizer, then call processor() for image processing + tokenization.
    text = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )
    result: dict[str, Any] = {}
    for key, value in inputs.items():
        result[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return result


def capture_task_hidden_states(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    sample: MemeSample,
    cfg: QwenResidualConfig,
) -> QwenTaskRun:
    """Run the Qwen model on a sample and capture residual stream hidden states.

    Uses the same prompt-building and capture strategy as the Gemma 3 pipeline,
    adapted for Qwen3.5-9B-Base's processor and model structure.
    """
    from PIL import Image

    from gemma3_experiments.residual_sae.datasets import build_classification_prompt

    prompt_text = build_classification_prompt(sample)
    image = Image.open(sample.image_path).convert("RGB")
    inputs = build_multimodal_inputs(processor, image, prompt_text, cfg.model_device)
    target_module = resolve_target_module(model, cfg)

    if cfg.capture_generation:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
                or processor.tokenizer.eos_token_id,
            )
        input_len = int(inputs["input_ids"].shape[1])
        new_tokens = output_ids[0, input_len:]
        generated_text = processor.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        if generated_text:
            full_inputs = build_multimodal_inputs(
                processor,
                image,
                prompt_text,
                cfg.model_device,
                assistant_text=generated_text,
                add_generation_prompt=False,
            )
            output_ids = full_inputs["input_ids"]
        else:
            full_inputs = inputs
            output_ids = full_inputs["input_ids"]
    else:
        output_ids = inputs["input_ids"]
        input_len = int(output_ids.shape[1])
        generated_text = ""
        full_inputs = inputs

    hidden_buf: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: Any, output: Any):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden_buf.append(hidden.detach().cpu())

    handle = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**full_inputs, use_cache=False)
    finally:
        handle.remove()

    assert hidden_buf, (
        f"Failed to capture hidden states for sample {sample.sample_id}"
    )
    hidden = hidden_buf[-1][0]
    full_input_ids = full_inputs["input_ids"][0].detach().cpu()
    assert full_input_ids.shape[0] == hidden.shape[0], (
        f"Token/hidden length mismatch for sample {sample.sample_id}: "
        f"{full_input_ids.shape[0]} != {hidden.shape[0]}"
    )
    image_span = _image_token_span(full_input_ids, processor, hidden.shape[0])
    return QwenTaskRun(
        hidden=hidden.to(torch.float32),
        output_ids=full_input_ids,
        input_len=input_len,
        image_span=image_span,
        generated_text=generated_text,
        prompt_text=prompt_text,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Token role helpers (mirrors gemma.py for multimodal token selections)
# ─────────────────────────────────────────────────────────────────────────────

ROLE_PROMPT = 0
ROLE_IMAGE = 1
ROLE_GENERATED = 2


def token_roles(n_tokens: int, image_span: tuple[int, int], input_len: int) -> torch.Tensor:
    """Assign role labels to token positions (0=prompt, 1=image, 2=generated)."""
    roles = torch.full((n_tokens,), ROLE_PROMPT, dtype=torch.int64)
    start, end = image_span
    roles[start:end] = ROLE_IMAGE
    if input_len < n_tokens:
        roles[input_len:] = ROLE_GENERATED
    return roles


def selection_mask(roles: torch.Tensor, selection: str) -> torch.Tensor:
    """Return a boolean mask for the requested token selection mode."""
    if selection == "all":
        return torch.ones_like(roles, dtype=torch.bool)
    if selection == "image":
        return roles == ROLE_IMAGE
    if selection == "prompt":
        return roles == ROLE_PROMPT
    if selection == "generated":
        return roles == ROLE_GENERATED
    if selection == "image_and_generated":
        return (roles == ROLE_IMAGE) | (roles == ROLE_GENERATED)
    raise ValueError(f"Unsupported token selection: {selection}")


# ─────────────────────────────────────────────────────────────────────────────
#  Constrained yes/no classification
# ─────────────────────────────────────────────────────────────────────────────

_YES_NO_IDS_CACHE: Optional[tuple[list[int], list[int]]] = None


def get_yes_no_token_ids(processor: Any) -> tuple[list[int], list[int]]:
    """Return single-token IDs for yes/no variants from the tokenizer."""
    global _YES_NO_IDS_CACHE
    if _YES_NO_IDS_CACHE is not None:
        return _YES_NO_IDS_CACHE

    tok = processor.tokenizer
    yes_ids: list[int] = []
    no_ids: list[int] = []

    for text in (" Yes", " yes", "Yes", "yes"):
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in yes_ids:
            yes_ids.append(ids[0])
    for text in (" No", " no", "No", "no"):
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in no_ids:
            no_ids.append(ids[0])

    if not yes_ids or not no_ids:
        raise RuntimeError(
            "Could not resolve single-token IDs for 'yes'/'no' in the tokenizer."
        )

    _YES_NO_IDS_CACHE = (yes_ids, no_ids)
    return _YES_NO_IDS_CACHE


class _YesNoLogitsProcessor:
    def __init__(self, allowed_ids: list[int]) -> None:
        self.allowed_ids = set(allowed_ids)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for tid in self.allowed_ids:
            mask[:, tid] = 0.0
        return scores + mask


def classify_constrained(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    image: Any,
    prompt_text: str,
    device: str,
    *,
    hook: Optional[Any] = None,
) -> tuple[bool, str, float]:
    """Classify image+text via yes/no logit scoring (constrained decoding)."""
    inputs = build_multimodal_inputs(processor, image, prompt_text, device)
    yes_ids, no_ids = get_yes_no_token_ids(processor)
    constraint = _YesNoLogitsProcessor(yes_ids + no_ids)

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.pad_token_id
            or processor.tokenizer.eos_token_id,
            logits_processor=[constraint],
        )

    logits_first = gen_out.scores[0][0].float().cpu()
    logit_yes = max(float(logits_first[i]) for i in yes_ids)
    logit_no = max(float(logits_first[i]) for i in no_ids)
    is_positive = logit_yes > logit_no
    conf_gap = float(torch.sigmoid(torch.tensor(logit_yes - logit_no)))
    return is_positive, ("yes" if is_positive else "no"), conf_gap


def score_label_candidates(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    image: Any,
    prompt_text: str,
    label_texts: list[str] | tuple[str, ...],
    device: str,
) -> tuple[str, dict[str, float], float]:
    pass
    assert label_texts, "Expected at least one candidate label."

    prompt_inputs = build_multimodal_inputs(processor, image, prompt_text, device)
    prompt_len = int(prompt_inputs["input_ids"].shape[1])

    label_scores: dict[str, float] = {}
    for label_text in label_texts:
        # Tokenize only the label text (no special tokens, no chat template).
        label_ids = processor.tokenizer(
            label_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)  # (1, label_len)
        label_len = int(label_ids.shape[1])
        assert label_len > 0, f"Candidate label {label_text!r} produced no tokens."

        # Extend input_ids and all per-token tensors (attention_mask,
        # mm_token_type_ids, etc.) by label_len positions.
        # New text tokens get type 0; attention mask gets 1.
        full_ids = torch.cat([prompt_inputs["input_ids"], label_ids], dim=1)
        full_inputs: dict[str, Any] = {}
        for k, v in prompt_inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape == (1, prompt_len):
                if k == "input_ids":
                    full_inputs[k] = full_ids
                else:
                    # attention_mask → pad with 1; all others (mm_token_type_ids, …) → pad with 0
                    pad_val = 1 if k == "attention_mask" else 0
                    pad = torch.full(
                        (1, label_len), pad_val,
                        dtype=v.dtype, device=device,
                    )
                    full_inputs[k] = torch.cat([v, pad], dim=1)
            else:
                full_inputs[k] = v

        with torch.no_grad():
            outputs = model(**full_inputs, use_cache=False)

        # Score the label tokens: logit at position (prompt_len - 1) predicts
        # token at prompt_len, etc.
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        start = prompt_len - 1
        stop = start + label_len
        token_logprobs = torch.log_softmax(logits[start:stop].float(), dim=-1)
        flat_label_ids = label_ids[0]
        label_score = token_logprobs[
            torch.arange(label_len, device=flat_label_ids.device), flat_label_ids
        ].sum()
        label_scores[label_text] = float(label_score.cpu())

    best_label = max(label_scores, key=label_scores.get)
    score_tensor = torch.tensor(
        [label_scores[label] for label in label_texts], dtype=torch.float32
    )
    confidence = float(torch.softmax(score_tensor, dim=0).max().item())
    return best_label, label_scores, confidence