import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from .config import GemmaResidualConfig
from .datasets import MemeSample, build_classification_prompt


BASE_IMG_TOKENS = 256


@dataclass
class GemmaTaskRun:
    hidden: torch.Tensor
    output_ids: torch.Tensor
    input_len: int
    image_span: tuple[int, int]
    generated_text: str
    prompt_text: str


@dataclass
class GemmaSaeConfig:
    d_in: int
    num_latents: int
    l0: int = 60
    architecture: str = "jump_relu"


class GemmaSae(nn.Module):
    def __init__(self, cfg: GemmaSaeConfig, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        self.num_latents = cfg.num_latents
        self.w_enc = nn.Parameter(torch.zeros(cfg.d_in, cfg.num_latents, device=device))
        self.b_enc = nn.Parameter(torch.zeros(cfg.num_latents, device=device))
        self.w_dec = nn.Parameter(torch.zeros(cfg.num_latents, cfg.d_in, device=device))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in, device=device))
        self.threshold = nn.Parameter(torch.zeros(cfg.num_latents, device=device))

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        sae_in = x.to(self.w_enc.dtype) - self.b_dec
        return sae_in @ self.w_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.pre_acts(x)
        return pre * (pre > self.threshold).to(pre.dtype)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.w_dec + self.b_dec

    def reconstruct(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        return acts, self.decode(acts)

    @staticmethod
    def load_from_hub(repo_id: str, hookpoint: str, device: str = "cpu") -> "GemmaSae":
        repo_path = Path(
            snapshot_download(
                repo_id,
                allow_patterns=[f"{hookpoint}/config.json", f"{hookpoint}/params.safetensors"],
            )
        ) / hookpoint
        with (repo_path / "config.json").open() as fd:
            cfg_json = json.load(fd)
        state_dict = load_file(str(repo_path / "params.safetensors"))
        cfg = GemmaSaeConfig(
            d_in=state_dict["w_enc"].shape[0],
            num_latents=state_dict["w_enc"].shape[1],
            l0=cfg_json.get("l0", 60),
        )
        sae = GemmaSae(cfg, device=device)
        sae.load_state_dict(state_dict, strict=True)
        sae.eval()
        for param in sae.parameters():
            param.requires_grad_(False)
        return sae


def load_gemma3_model(cfg: GemmaResidualConfig) -> tuple[Gemma3ForConditionalGeneration, Any]:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map={"": cfg.model_device},
    ).eval()
    return model, processor


def resolve_target_module(model: Gemma3ForConditionalGeneration, cfg: GemmaResidualConfig) -> nn.Module:
    candidates = [
        cfg.hook_module_path,
        "model.layers.22",
        "language_model.layers.22",
        "language_model.model.layers.22",
    ]
    for path in candidates:
        try:
            return model.get_submodule(path)
        except AttributeError:
            continue
    raise RuntimeError(f"Could not find a Gemma 3 residual stream module. Tried: {candidates}")


def _image_token_span(input_ids: torch.Tensor, processor: Any, seq_len: int) -> tuple[int, int]:
    image_seq_len = getattr(processor, "image_seq_length", BASE_IMG_TOKENS)
    image_tok_id = None
    for attr in ("image_token_id", "image_token"):
        value = getattr(processor, attr, None)
        if value is None:
            continue
        if isinstance(value, str):
            value = processor.tokenizer.convert_tokens_to_ids(value)
        if isinstance(value, int) and value != processor.tokenizer.unk_token_id:
            image_tok_id = value
            break
    if image_tok_id is not None:
        positions = (input_ids == image_tok_id).nonzero(as_tuple=False)
        if positions.numel() > 0:
            start = int(positions[0].item())
            return start, min(start + image_seq_len, seq_len)
    ids = input_ids.tolist()
    best_start = 0
    best_len = 0
    best_tok = -1
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
    image,
    prompt_text: str,
    device: str,
    *,
    assistant_text: str = "",
    add_generation_prompt: bool = True,
) -> dict[str, Any]:
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
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    result: dict[str, Any] = {}
    for key, value in inputs.items():
        result[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return result


def capture_task_hidden_states(
    model: Gemma3ForConditionalGeneration,
    processor: Any,
    sample: MemeSample,
    cfg: GemmaResidualConfig,
    *,
    prompt_override: str | None = None,
) -> GemmaTaskRun:
    from PIL import Image

    prompt_text = prompt_override if prompt_override is not None else build_classification_prompt(sample)
    image = Image.open(sample.image_path).convert("RGB")
    inputs = build_multimodal_inputs(processor, image, prompt_text, cfg.model_device)
    target_module = resolve_target_module(model, cfg)

    if cfg.capture_generation:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )
        input_len = int(inputs["input_ids"].shape[1])
        new_tokens = output_ids[0, input_len:]
        generated_text = processor.decode(new_tokens, skip_special_tokens=True).strip()
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

    assert hidden_buf, f"Failed to capture hidden states for sample {sample.sample_id}"
    hidden = hidden_buf[-1][0]
    full_input_ids = full_inputs["input_ids"][0].detach().cpu()
    assert full_input_ids.shape[0] == hidden.shape[0], (
        f"Token/hidden length mismatch for sample {sample.sample_id}: "
        f"{full_input_ids.shape[0]} != {hidden.shape[0]}"
    )
    image_span = _image_token_span(full_input_ids, processor, hidden.shape[0])
    return GemmaTaskRun(
        hidden=hidden.to(torch.float32),
        output_ids=full_input_ids,
        input_len=input_len,
        image_span=image_span,
        generated_text=generated_text,
        prompt_text=prompt_text,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Constrained yes/no classification
# ─────────────────────────────────────────────────────────────────────────────

_YES_NO_IDS_CACHE: Optional[tuple[list[int], list[int]]] = None


def get_yes_no_token_ids(processor: Any) -> tuple[list[int], list[int]]:
    """Return single-token IDs for yes/no variants from the tokenizer.

    Results are cached after the first call so repeated calls are free.
    Raises RuntimeError if no single-token yes/no IDs can be found.
    """
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
            "Could not resolve single-token IDs for 'yes'/'no' in the tokenizer. "
            "Check that the tokenizer has the expected vocabulary."
        )

    _YES_NO_IDS_CACHE = (yes_ids, no_ids)
    return _YES_NO_IDS_CACHE


class _YesNoLogitsProcessor:
    """Logits processor that masks every token except yes/no token IDs.

    Guarantees the model can only output a yes or no token, eliminating
    free-generation collapse under SAE interventions.
    """

    def __init__(self, allowed_ids: list[int]) -> None:
        self.allowed_ids = set(allowed_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for tid in self.allowed_ids:
            mask[:, tid] = 0.0
        return scores + mask


def classify_constrained(
    model: Gemma3ForConditionalGeneration,
    processor: Any,
    image: Any,
    prompt_text: str,
    device: str,
    *,
    hook: Optional[Any] = None,
) -> tuple[bool, str, float]:
    pass
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
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            logits_processor=[constraint],
        )

    logits_first = gen_out.scores[0][0].float().cpu()
    logit_yes = max(float(logits_first[i]) for i in yes_ids)
    logit_no = max(float(logits_first[i]) for i in no_ids)
    is_positive = logit_yes > logit_no
    conf_gap = float(torch.sigmoid(torch.tensor(logit_yes - logit_no)))
    return is_positive, ("yes" if is_positive else "no"), conf_gap


def score_label_candidates(
    model: Gemma3ForConditionalGeneration,
    processor: Any,
    image: Any,
    prompt_text: str,
    label_texts: list[str] | tuple[str, ...],
    device: str,
) -> tuple[str, dict[str, float], float]:
    """Score candidate label strings as the assistant response.

    Returns `(best_label, logprob_by_label, confidence)` where confidence is the
    softmax probability of the best-scoring label across the candidate set.
    """
    assert label_texts, "Expected at least one candidate label."

    prompt_inputs = build_multimodal_inputs(processor, image, prompt_text, device)
    prompt_ids = prompt_inputs["input_ids"][0]
    prompt_len = int(prompt_ids.shape[0])

    label_scores: dict[str, float] = {}
    for label_text in label_texts:
        full_inputs = build_multimodal_inputs(
            processor,
            image,
            prompt_text,
            device,
            assistant_text=label_text,
            add_generation_prompt=False,
        )
        full_ids = full_inputs["input_ids"][0]
        assert torch.equal(full_ids[:prompt_len], prompt_ids), (
            "Prompt prefix changed while building label-scoring inputs. "
            f"label={label_text!r}"
        )
        label_ids = full_ids[prompt_len:]
        assert len(label_ids) > 0, f"Candidate label {label_text!r} produced no assistant tokens."

        with torch.no_grad():
            outputs = model(**full_inputs, use_cache=False)

        logits = outputs.logits[0]
        start = prompt_len - 1
        stop = start + len(label_ids)
        token_logprobs = torch.log_softmax(logits[start:stop].float(), dim=-1)
        label_score = token_logprobs[
            torch.arange(len(label_ids), device=label_ids.device),
            label_ids,
        ].sum()
        label_scores[label_text] = float(label_score.cpu())

    best_label = max(label_scores, key=label_scores.get)
    score_tensor = torch.tensor([label_scores[label] for label in label_texts], dtype=torch.float32)
    confidence = float(torch.softmax(score_tensor, dim=0).max().item())
    return best_label, label_scores, confidence
