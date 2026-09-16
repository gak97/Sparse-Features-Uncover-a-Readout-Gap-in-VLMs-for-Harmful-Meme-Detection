

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from simple_parsing import Serializable, list_field, parse

from ..config import QwenResidualConfig
from ..qwen import (
    QwenSae,
    _image_token_span,
    build_multimodal_inputs,
    classify_constrained,
    load_qwen3_model,
)
from ..task_labels import fit_scaled_logreg
from .direct_logit_projection import _extract_features_with_ids, _read_jsonl

logger = logging.getLogger(__name__)

ROLE_ORDER = ("image", "prompt", "answer")
WRONG_ROLE = {"image": "prompt", "prompt": "image", "answer": "image"}


@dataclass(frozen=True)
class TaskSpec:
    dataset_name: str
    task_name: str
    positive_label: str
    train_dir: str
    eval_dir: str
    train_split: str
    eval_split: str
    baseline_path: str


def _specs(root: str) -> dict[str, TaskSpec]:
    dense = f"{root}/qwen_residual_sae_cache"
    return {
        "fhm": TaskSpec(
            "Hateful_Memes_Dataset", "hateful_memes_binary", "hateful",
            f"{dense}/dense", f"{dense}/dense_val_test", "train", "test",
            f"{root}/qwen_infer/baseline_proper.jsonl",
        ),
        "mami": TaskSpec(
            "MAMI", "misogyny_and_subtypes", "misogynous",
            f"{dense}/dense", f"{dense}/dense_val_test", "train", "test",
            f"{root}/qwen_infer/baseline_proper.jsonl",
        ),
        "multibully": TaskSpec(
            "MultiBully", "cyberbullying_detection", "bully",
            f"{root}/multibully/dense_raw_v2", f"{root}/multibully/dense_raw_v2", "probe_train", "eval",
            f"{root}/multibully/baseline_raw_ocr_v2/baseline_eval.jsonl",
        ),
    }


@dataclass
class LayerSelfPatchConfig(Serializable):
    dataset: str = "fhm"
    activations_root: str = "./outputs"
    output_dir: str = "./outputs/layer_self_patching/fhm"
    model_id: str = "Qwen/Qwen3.5-9B-Base"
    sae_repo_id: str = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
    sae_layer: int = 20
    device: str = "cuda:0"
    C: float = 1.0
    max_iter: int = 3000
    n_examples: int = 80
    top_k_features: int = 20
    layer_pairs: list[str] = list_field("16:20", "20:24", "24:28", "20:30")
    recoding_layers: list[int] = list_field(16, 20, 24, 28, 30)
    recoding_freeze_layers: list[int] = list_field(20, 28)
    live_screen_baseline: bool = False
    seed: int = 20260809
    resume: bool = True


def _load_rows(directory: Path, spec: TaskSpec, split: str) -> list[dict]:
    return [
        row for row in _read_jsonl(directory / "manifest.jsonl")
        if row.get("dataset_name") == spec.dataset_name
        and row.get("task_name") == spec.task_name
        and row.get("split") == split
        and row.get("gold_fields", {}).get("label") is not None
    ]


def _labels(rows: list[dict], positive_label: str) -> np.ndarray:
    return np.asarray(
        [int(str(row["gold_fields"]["label"]).lower() == positive_label.lower()) for row in rows],
        dtype=np.int32,
    )


def _logit_from_confidence(confidence: float) -> float:
    probability = min(max(float(confidence), 1e-6), 1 - 1e-6)
    return math.log(probability / (1 - probability))


def _baseline_logits(path: Path, spec: TaskSpec) -> dict[str, float]:
    result = {}
    for row in _read_jsonl(path):
        if path.name == "baseline_eval.jsonl" or (
            row.get("dataset_name") == spec.dataset_name and row.get("task_name") == spec.task_name
        ):
            if row.get("conf_gap") is not None:
                result[str(row["sample_id"])] = _logit_from_confidence(row["conf_gap"])
    return result


def _role_positions(input_ids: torch.Tensor, processor) -> dict[str, torch.Tensor]:
    sequence_length = int(input_ids.shape[1])
    image_start, image_end = _image_token_span(input_ids[0].detach().cpu(), processor, sequence_length)
    image = torch.arange(image_start, image_end, dtype=torch.long)
    answer = torch.tensor([sequence_length - 1], dtype=torch.long)
    is_image = torch.zeros(sequence_length, dtype=torch.bool)
    is_image[image] = True
    is_image[answer] = True
    prompt = torch.arange(sequence_length, dtype=torch.long)[~is_image]
    return {"image": image, "prompt": prompt, "answer": answer}


def _make_inputs(processor, row: dict, device: str, prompt_text: str | None = None):
    image = Image.open(row["image_path"]).convert("RGB")
    inputs = build_multimodal_inputs(processor, image, prompt_text or row["prompt_text"], device)
    return inputs, _role_positions(inputs["input_ids"], processor)


def _capture_layers(model, inputs: dict, layers: list[int]) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in sorted(set(layers)):
        module = model.get_submodule(f"model.language_model.layers.{layer}")

        def hook(_module, _inputs, output, layer=layer):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden.detach()[0].to(torch.float32).cpu()

        handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(layers) - set(captured):
        raise RuntimeError(f"Did not capture requested layers: {set(layers) - set(captured)}")
    return captured


def _orthonormal_decoder_subspace(sae: QwenSae, feature_indices: np.ndarray, device: str) -> torch.Tensor:
    directions = sae.W_dec[:, torch.as_tensor(feature_indices, device=sae.W_dec.device)].float()
    basis, _ = torch.linalg.qr(directions, mode="reduced")
    return basis.to(device)


def _random_subspace(d_model: int, rank: int, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    basis, _ = torch.linalg.qr(torch.randn(d_model, rank, generator=generator), mode="reduced")
    return basis.to(device)


def _project(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (values @ basis) @ basis.T


def _aligned_indices(target: torch.Tensor, source: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n_positions = min(len(target), len(source))
    return target[:n_positions].to(device), source[:n_positions].to(device)


def _patch_hook(
    sae: QwenSae,
    source_layer: int,
    target_layer: int,
    target_positions: torch.Tensor,
    source_positions: torch.Tensor,
    basis: torch.Tensor,
    condition: str,
    feature_indices: np.ndarray,
    donor_source: torch.Tensor | None = None,
):
    source_state: torch.Tensor | None = donor_source

    def source_hook(_module, _inputs, output):
        nonlocal source_state
        hidden = output[0] if isinstance(output, tuple) else output
        source_state = hidden.detach()[0].to(torch.float32)

    def target_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if source_state is None:
            raise RuntimeError("Target layer ran before source state was captured")
        target = hidden.detach()[0].to(torch.float32)
        target_idx, source_idx = _aligned_indices(target_positions, source_positions, target.device)
        # A cached unrelated/neutral donor lives on CPU, while live source
        # states are on CUDA. Index on the source tensor's device, then move
        # only the selected natural-position values to the target device.
        source_idx = source_idx.to(source_state.device)
        updated = target.clone()
        if condition == "self_source_control":
            source_values = target[target_idx]
        else:
            source_values = source_state[source_idx].to(target.device)
        target_values = target[target_idx]
        if condition == "reconstruction_error_preserving":
            feature_idx = torch.as_tensor(feature_indices, dtype=torch.long, device=target.device)
            target_acts, _ = sae.encode(target_values)
            source_acts, _ = sae.encode(source_values)
            modified_acts = target_acts.clone()
            modified_acts[:, feature_idx] = source_acts[:, feature_idx]
            delta = sae.decode(modified_acts) - sae.decode(target_acts)
        else:
            delta = _project(source_values - target_values, basis)
            if condition in {"random_subspace", "low_probe_active"}:
                # Match the top-subspace perturbation magnitude at every natural
                # token position; only the subspace identity is changed.
                reference = _project(source_values - target_values, top_basis)
                scale = reference.norm(dim=-1, keepdim=True) / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                delta = delta * scale
        updated[target_idx] = target_values + delta
        updated = updated.unsqueeze(0).to(hidden.dtype).contiguous()
        return (updated, *output[1:]) if isinstance(output, tuple) else updated

    # Stored as an attribute so the caller can retrieve the exact source hook
    # without creating a separate closure for every intervention.
    target_hook.source_hook = None if donor_source is not None else source_hook  # type: ignore[attr-defined]
    return target_hook


# Set per-example before constructing `_patch_hook`; this lets random and
# low-probe controls match the natural top-probe perturbation norm exactly.
top_basis: torch.Tensor


def _score_with_patch(model, processor, inputs: dict, source_layer: int, target_layer: int, hook, device: str) -> float:
    source_module = model.get_submodule(f"model.language_model.layers.{source_layer}")
    target_module = model.get_submodule(f"model.language_model.layers.{target_layer}")
    handles = []
    if source_layer < target_layer and getattr(hook, "source_hook", None) is not None:
        handles.append(source_module.register_forward_hook(hook.source_hook))
    handles.append(target_module.register_forward_hook(hook))
    try:
        image_tensor = inputs.pop("_audit_image")
        prompt_text = inputs.pop("_audit_prompt")
        _prediction, _text, confidence = classify_constrained(model, processor, image_tensor, prompt_text, device)
        return _logit_from_confidence(confidence)
    finally:
        inputs["_audit_image"] = image_tensor
        inputs["_audit_prompt"] = prompt_text
        for handle in handles:
            handle.remove()


def _score_baseline(model, processor, image: Image.Image, prompt_text: str, device: str) -> float:
    _prediction, _text, confidence = classify_constrained(model, processor, image, prompt_text, device)
    return _logit_from_confidence(confidence)


def _neutral_prompt(row: dict) -> str:
    ocr = str(row.get("raw_fields", {}).get("ocr_text", row.get("meme_text", "")))
    return (
        "Describe this meme's visible people, objects, layout, and text. "
        "Transcribe any readable text. Do not assess hatefulness, harmfulness, or labels.\n\n"
        f"Meme text: {ocr}"
    )


def _existing_keys(path: Path) -> set[tuple[str, str, int, int, str]]:
    if not path.exists():
        return set()
    keys = set()
    for row in _read_jsonl(path):
        condition = str(row.get("condition", ""))
        if row.get("record_type") == "self_patch":
            condition = f"{row.get('role', '')}:{condition}"
        keys.add((
            str(row["sample_id"]), str(row["record_type"]),
            int(row.get("source_layer", -1)), int(row.get("target_layer", -1)), condition,
        ))
    return keys


def _select_examples(rows: list[dict], labels: np.ndarray, scores: np.ndarray, baseline_logits: dict[str, float], n_examples: int) -> list[int]:
    candidates = [
        index for index, row in enumerate(rows)
        if str(row["sample_id"]) in baseline_logits
        and int(baseline_logits[str(row["sample_id"])] > 0) != int(labels[index])
        and int(scores[index] > 0) == int(labels[index])
    ]
    by_label = {label: sorted((index for index in candidates if labels[index] == label), key=lambda index: -abs(scores[index])) for label in (0, 1)}
    chosen: list[int] = []
    per_label = n_examples // 2
    for label in (0, 1):
        chosen.extend(by_label[label][:per_label])
    remainder = sorted(set(candidates) - set(chosen), key=lambda index: -abs(scores[index]))
    chosen.extend(remainder[: n_examples - len(chosen)])
    return chosen


def _low_probe_active_features(train_features: np.ndarray, coefficient: np.ndarray, top_features: np.ndarray, top_k: int) -> np.ndarray:
    activity = train_features.mean(axis=0)
    target_activity = float(activity[top_features].mean())
    low_weight = np.abs(coefficient) <= np.quantile(np.abs(coefficient), 0.25)
    candidates = np.flatnonzero(low_weight)
    candidates = candidates[~np.isin(candidates, top_features)]
    return candidates[np.argsort(np.abs(activity[candidates] - target_activity))[:top_k]]


def _summary(records: list[dict]) -> dict:
    grouped: dict[tuple[str, int, int, str, str], list[dict]] = {}
    for row in records:
        if row.get("record_type") != "self_patch":
            continue
        key = (row["dataset"], row["source_layer"], row["target_layer"], row["role"], row["condition"])
        grouped.setdefault(key, []).append(row)
    output = {}
    for key, rows in grouped.items():
        margins = np.asarray([row["delta_gold_margin"] for row in rows])
        correctness = np.asarray([row["delta_correct"] for row in rows])
        output["/".join(map(str, key))] = {
            "n": len(rows),
            "mean_delta_gold_margin": round(float(margins.mean()), 4),
            "correctness_flip_rate": round(float(correctness.mean()), 4),
            "n_correctness_improved": int((correctness > 0).sum()),
            "n_correctness_harmed": int((correctness < 0).sum()),
        }
    return output


def _recoding_summary(records: list[dict]) -> dict:
    projections: dict[int, list[dict]] = {}
    freezes: dict[tuple[int, str], list[dict]] = {}
    for row in records:
        if row.get("record_type") == "visual_recoding_projection":
            projections.setdefault(int(row["target_layer"]), []).append(row)
        elif row.get("record_type") == "visual_recoding_freeze":
            freezes.setdefault((int(row["target_layer"]), str(row["condition"])), []).append(row)
    projection_summary = {
        str(layer): {
            "n": len(rows),
            "mean_harmful_minus_neutral_image_projection": round(
                float(np.mean([row["delta_harmful_minus_neutral"] for row in rows])), 4
            ),
        }
        for layer, rows in projections.items()
    }
    freeze_summary = {
        f"{layer}/{condition}": {
            "n": len(rows),
            "mean_delta_gold_margin": round(float(np.mean([row["delta_gold_margin"] for row in rows])), 4),
            "correctness_flip_rate": round(float(np.mean([
                row["patched_correct"] - row["baseline_correct"] for row in rows
            ])), 4),
        }
        for (layer, condition), rows in freezes.items()
    }
    return {"image_projection": projection_summary, "neutral_visual_freeze": freeze_summary}


def main() -> None:
    global top_basis
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = parse(LayerSelfPatchConfig)
    specs = _specs(cfg.activations_root)
    if cfg.dataset not in specs:
        raise ValueError(f"Unknown dataset {cfg.dataset!r}; choose from {sorted(specs)}")
    spec = specs[cfg.dataset]
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.jsonl"
    existing = _existing_keys(output_path) if cfg.resume else set()

    train_rows = _load_rows(Path(spec.train_dir), spec, spec.train_split)
    eval_rows = _load_rows(Path(spec.eval_dir), spec, spec.eval_split)
    if not train_rows or not eval_rows:
        raise ValueError(f"Missing train/eval rows: {len(train_rows)}/{len(eval_rows)}")
    sae = QwenSae.load_from_hub(cfg.sae_repo_id, cfg.sae_layer, device=cfg.device)
    train_features, train_labels_raw, _ = _extract_features_with_ids(sae, Path(spec.train_dir), train_rows, "label", "max", cfg.device)
    eval_features, eval_labels_raw, eval_ids = _extract_features_with_ids(sae, Path(spec.eval_dir), eval_rows, "label", "max", cfg.device)
    train_labels = np.asarray([int(label.lower() == spec.positive_label.lower()) for label in train_labels_raw], dtype=np.int32)
    eval_labels = np.asarray([int(label.lower() == spec.positive_label.lower()) for label in eval_labels_raw], dtype=np.int32)
    classifier, scaler = fit_scaled_logreg(train_features, train_labels, C=cfg.C, max_iter=cfg.max_iter)
    eval_scores = classifier.decision_function(scaler.transform(eval_features))
    coefficient = classifier.coef_[0].copy()
    if list(classifier.classes_)[1] != 1:
        coefficient *= -1
    top_features = np.argsort(-np.abs(coefficient))[:cfg.top_k_features]
    low_features = _low_probe_active_features(train_features, coefficient, top_features, cfg.top_k_features)
    baseline_logits = _baseline_logits(Path(spec.baseline_path), spec)
    eval_row_by_id = {str(row["sample_id"]): row for row in eval_rows}
    ordered_eval_rows = [eval_row_by_id[str(sample_id)] for sample_id in eval_ids]
    if len(eval_row_by_id) != len(eval_rows) or any(
        str(row["sample_id"]) != str(sample_id)
        for row, sample_id in zip(ordered_eval_rows, eval_ids)
    ):
        raise ValueError("Evaluation manifest rows do not uniquely align to extracted feature sample IDs")
    historical_candidates = _select_examples(
        ordered_eval_rows,
        eval_labels,
        eval_scores,
        baseline_logits,
        len(eval_ids) if cfg.live_screen_baseline else cfg.n_examples,
    )
    if not historical_candidates:
        raise ValueError("No base-wrong/probe-right examples were available")
    if not cfg.live_screen_baseline and any(
        int(baseline_logits[str(eval_ids[index])] > 0) == int(eval_labels[index])
        or int(eval_scores[index] > 0) != int(eval_labels[index])
        for index in historical_candidates
    ):
        raise AssertionError("Selected examples must be base-wrong and probe-right under aligned IDs")

    qwen_cfg = QwenResidualConfig(model_id=cfg.model_id, model_device=cfg.device, sae_layer=cfg.sae_layer)
    model, processor = load_qwen3_model(qwen_cfg)
    live_baseline_logits: dict[str, float] = {}
    if cfg.live_screen_baseline:
        selected = []
        for index in historical_candidates:
            row = ordered_eval_rows[index]
            image = Image.open(row["image_path"]).convert("RGB")
            logit = _score_baseline(model, processor, image, row["prompt_text"], cfg.device)
            live_baseline_logits[str(eval_ids[index])] = logit
            if int(logit > 0) != int(eval_labels[index]):
                selected.append(index)
                if len(selected) == cfg.n_examples:
                    break
        if not selected:
            raise ValueError("No live-base-wrong/probe-right examples were available")
        logger.info(
            "%s: live screening retained %d/%d historical candidates",
            cfg.dataset, len(selected), len(historical_candidates),
        )
    else:
        selected = historical_candidates
    (output_dir / "selection.json").write_text(json.dumps({
        "dataset": cfg.dataset, "n_selected": len(selected), "n_requested": cfg.n_examples,
        "selection": "live_base_wrong_probe_right" if cfg.live_screen_baseline else "base_wrong_probe_right",
        "n_historical_candidates": len(historical_candidates),
        "top_probe_features": top_features.tolist(),
        "low_probe_equally_active_features": low_features.tolist(),
        "sample_ids": [str(eval_ids[index]) for index in selected],
    }, indent=2) + "\n")
    logger.info("%s: selected %d base-wrong/probe-right examples", cfg.dataset, len(selected))
    top_basis = _orthonormal_decoder_subspace(sae, top_features, cfg.device)
    low_basis = _orthonormal_decoder_subspace(sae, low_features, cfg.device)
    random_basis = _random_subspace(sae.cfg.d_model, top_basis.shape[1], cfg.seed, cfg.device)
    layer_pairs = [tuple(map(int, pair.split(":"))) for pair in cfg.layer_pairs]
    if any(source >= target for source, target in layer_pairs):
        raise ValueError("Each layer pair must satisfy source < target")

    # Two fixed unrelated donors make the donor control deterministic without
    # repeatedly loading a separate donor for every target example.
    donor_indices = selected[:2]
    donor_cache: dict[tuple[int, int], tuple[torch.Tensor, dict[str, torch.Tensor]]] = {}
    records: list[dict] = []
    with output_path.open("a") as fd:
        for sample_number, index in enumerate(selected, start=1):
            row = ordered_eval_rows[index]
            sample_id = str(row["sample_id"])
            image = Image.open(row["image_path"]).convert("RGB")
            prompt = row["prompt_text"]
            inputs, roles = _make_inputs(processor, row, cfg.device)
            baseline_logit = live_baseline_logits.get(sample_id)
            if baseline_logit is None:
                baseline_logit = _score_baseline(model, processor, image, prompt, cfg.device)
            baseline_correct = int((baseline_logit > 0) == bool(eval_labels[index]))
            donor_index = next((candidate for candidate in donor_indices if candidate != index), index)
            donor_row = ordered_eval_rows[donor_index]

            for source_layer, target_layer in layer_pairs:
                donor_key = (donor_index, source_layer)
                if donor_key not in donor_cache:
                    donor_inputs, donor_roles = _make_inputs(processor, donor_row, cfg.device)
                    donor_state = _capture_layers(model, donor_inputs, [source_layer])[source_layer]
                    donor_cache[donor_key] = (donor_state, donor_roles)
                donor_state, donor_roles = donor_cache[donor_key]

                for role in ROLE_ORDER:
                    conditions = (
                        ("top_probe_subspace", top_basis, role, None),
                        ("self_source_control", top_basis, role, None),
                        ("random_subspace", random_basis, role, None),
                        ("low_probe_active", low_basis, role, None),
                        ("unrelated_example_donor", top_basis, role, donor_state),
                        ("wrong_token_role", top_basis, WRONG_ROLE[role], None),
                        ("reconstruction_error_preserving", top_basis, role, None),
                    )
                    for condition, basis, source_role, donor_source in conditions:
                        key = (sample_id, "self_patch", source_layer, target_layer, f"{role}:{condition}")
                        if key in existing:
                            continue
                        source_positions = donor_roles[source_role] if donor_source is not None else roles[source_role]
                        hook = _patch_hook(
                            sae, source_layer, target_layer, roles[role], source_positions, basis,
                            condition, top_features, donor_source=donor_source,
                        )
                        patched_inputs = {**inputs, "_audit_image": image, "_audit_prompt": prompt}
                        patched_logit = _score_with_patch(
                            model, processor, patched_inputs, source_layer, target_layer, hook, cfg.device,
                        )
                        gold = int(eval_labels[index])
                        patched_correct = int((patched_logit > 0) == bool(gold))
                        record = {
                            "record_type": "self_patch", "dataset": cfg.dataset, "sample_id": sample_id,
                            "gold": gold, "source_layer": source_layer, "target_layer": target_layer,
                            "role": role, "source_role": source_role, "condition": condition,
                            "baseline_logit": baseline_logit, "patched_logit": patched_logit,
                            "baseline_gold_margin": baseline_logit if gold else -baseline_logit,
                            "patched_gold_margin": patched_logit if gold else -patched_logit,
                            "delta_gold_margin": (patched_logit - baseline_logit) if gold else (baseline_logit - patched_logit),
                            "baseline_correct": baseline_correct, "patched_correct": patched_correct,
                            "delta_correct": patched_correct - baseline_correct,
                            "donor_sample_id": str(donor_row["sample_id"]) if donor_source is not None else None,
                            "reconstruction_error_preserved": condition == "reconstruction_error_preserving",
                        }
                        fd.write(json.dumps(record) + "\n")
                        fd.flush()
                        records.append(record)

            # Linguistic visual recoding is FHM-specific: capture the same
            # meme under the harmfulness versus neutral-description prompt,
            # then freeze neutral visual components into the harmfulness run.
            if cfg.dataset == "fhm":
                neutral = _neutral_prompt(row)
                neutral_inputs, neutral_roles = _make_inputs(processor, row, cfg.device, neutral)
                harm_states = _capture_layers(model, inputs, cfg.recoding_layers)
                neutral_states = _capture_layers(model, neutral_inputs, cfg.recoding_layers)
                for layer in cfg.recoding_layers:
                    key = (sample_id, "visual_recoding_projection", layer, layer, "harmful_vs_neutral")
                    if key in existing:
                        continue
                    harm_image = harm_states[layer][roles["image"]].to(cfg.device)
                    neutral_image = neutral_states[layer][neutral_roles["image"]].to(cfg.device)
                    harm_projection = _project(harm_image, top_basis).norm(dim=-1).mean().item()
                    neutral_projection = _project(neutral_image, top_basis).norm(dim=-1).mean().item()
                    record = {
                        "record_type": "visual_recoding_projection", "dataset": cfg.dataset,
                        "sample_id": sample_id, "source_layer": layer, "target_layer": layer,
                        "condition": "harmful_vs_neutral", "harmful_image_projection_norm": harm_projection,
                        "neutral_image_projection_norm": neutral_projection,
                        "delta_harmful_minus_neutral": harm_projection - neutral_projection,
                    }
                    fd.write(json.dumps(record) + "\n")
                    fd.flush()
                    records.append(record)
                for layer in cfg.recoding_freeze_layers:
                    for condition, basis, donor_state in (
                        ("neutral_visual_freeze_top_subspace", top_basis, neutral_states[layer]),
                        ("harmful_visual_self_control", top_basis, harm_states[layer]),
                        ("neutral_visual_freeze_random_subspace", random_basis, neutral_states[layer]),
                    ):
                        key = (sample_id, "visual_recoding_freeze", layer, layer, condition)
                        if key in existing:
                            continue
                        hook = _patch_hook(
                            sae, layer - 1, layer, roles["image"], neutral_roles["image"], basis,
                            "top_probe_subspace", top_features, donor_source=donor_state,
                        )
                        patched_inputs = {**inputs, "_audit_image": image, "_audit_prompt": prompt}
                        patched_logit = _score_with_patch(model, processor, patched_inputs, layer - 1, layer, hook, cfg.device)
                        gold = int(eval_labels[index])
                        record = {
                            "record_type": "visual_recoding_freeze", "dataset": cfg.dataset,
                            "sample_id": sample_id, "gold": gold, "source_layer": layer, "target_layer": layer,
                            "role": "image", "condition": condition, "baseline_logit": baseline_logit,
                            "patched_logit": patched_logit,
                            "delta_gold_margin": (patched_logit - baseline_logit) if gold else (baseline_logit - patched_logit),
                            "baseline_correct": baseline_correct,
                            "patched_correct": int((patched_logit > 0) == bool(gold)),
                        }
                        fd.write(json.dumps(record) + "\n")
                        fd.flush()
                        records.append(record)
            if sample_number % 5 == 0 or sample_number == len(selected):
                logger.info("%s progress: %d/%d", cfg.dataset, sample_number, len(selected))

    all_records = _read_jsonl(output_path)
    summary = {
        "dataset": cfg.dataset,
        "n_selected": len(selected),
        "layer_pairs": cfg.layer_pairs,
        "roles": ROLE_ORDER,
        "conditions": [
            "top_probe_subspace", "self_source_control", "random_subspace", "low_probe_active",
            "unrelated_example_donor", "wrong_token_role", "reconstruction_error_preserving",
        ],
        "self_patch": _summary(all_records),
    }
    if cfg.dataset == "fhm":
        projections = [row for row in all_records if row.get("record_type") == "visual_recoding_projection"]
        freezes = [row for row in all_records if row.get("record_type") == "visual_recoding_freeze"]
        summary["visual_recoding"] = {
            "n_projection_records": len(projections), "n_freeze_records": len(freezes),
            **_recoding_summary(all_records),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote layer self-patching results to %s", output_dir)


if __name__ == "__main__":
    main()