

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from simple_parsing import ArgumentGenerationMode, parse


from gemma3_experiments.launch.residual.probe_fhm_crosscoder import GemmaCrosscoder
from gemma3_experiments.launch.residual.probe_validate import (
    _load_fhm_confounder_metadata,
    _normalise_fhm_id,
    _normalise_group_value,
    _per_class_metrics,
)
from gemma3_experiments.residual_sae import (
    GemmaResidualConfig,
    MemeDatasetConfig,
    MemeSample,
    build_multimodal_inputs,
    build_simple_classify_prompt,
    get_yes_no_token_ids,
    load_gemma3_model,
    load_meme_samples,
    resolve_visible_device,
)
from gemma3_experiments.residual_sae.gemma import _image_token_span

logger = logging.getLogger(__name__)


@dataclass
class FhmLayerSelfPatchingConfig:
    dataset: MemeDatasetConfig = field(default_factory=lambda: MemeDatasetConfig(
        datasets=["Hateful_Memes_Dataset"], splits=["test"], fhm_label_split="test_seen",
    ))
    gemma: GemmaResidualConfig = field(default_factory=lambda: GemmaResidualConfig(
        model_id="google/gemma-3-12b-it", model_device="cuda:0",
    ))
    checkpoint_dir: str = ""
    crosscoder_repo_id: str = "google/gemma-scope-2-12b-it"
    crosscoder_hookpoint: str = "resid_post/layer_31_width_65k_l0_medium"
    source_module_path: str = "model.language_model.layers.31"
    target_module_path: str = "model.language_model.layers.41"
    fhm_confounder_metadata_path: str = ""
    output_dir: str = "./fhm_layer_self_patching"

    max_samples: int = 64
    seed: int = 42
    low_weight_candidate_features: int = 512
    low_weight_calibration_samples: int = 12
    device: str = "cuda:0"


def _output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_output(output: Any, hidden: torch.Tensor) -> Any:
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def _role_positions(input_ids: torch.Tensor, processor: Any) -> dict[str, torch.Tensor]:
    image_start, image_stop = _image_token_span(input_ids, processor, int(input_ids.shape[0]))
    image = torch.arange(image_start, image_stop, device=input_ids.device)
    prompt_mask = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    prompt_mask[image_start:image_stop] = False
    return {"image": image, "prompt": torch.arange(input_ids.shape[0], device=input_ids.device)[prompt_mask]}


def _resample_positions(hidden: torch.Tensor, positions: torch.Tensor, n_positions: int) -> torch.Tensor:
    values = hidden[positions]
    assert len(values) > 0, "Cannot patch an empty token role."
    if len(values) == n_positions:
        return values
    indices = torch.linspace(0, len(values) - 1, n_positions, device=hidden.device).round().long()
    return values[indices]


def _orthonormal_columns(vectors: torch.Tensor, rank: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(vectors.float(), mode="reduced")
    assert q.shape[1] >= rank, f"Factor directions have rank {q.shape[1]}, expected {rank}."
    return q[:, :rank]


def _encoder_vectors(crosscoder: GemmaCrosscoder, feature_indices: list[int]) -> torch.Tensor:
    weights = crosscoder.w_enc
    indices = torch.tensor(feature_indices, device=weights.device)
    if weights.ndim == 3:
        assert weights.shape[0] == 1, f"Expected one layer, found encoder shape {tuple(weights.shape)}"
        weights = weights[0]
    if weights.shape[0] == crosscoder.d_model:
        return weights[:, indices].float()
    if weights.shape[1] == crosscoder.d_model:
        return weights[indices, :].T.float()
    raise ValueError(f"Cannot locate d_model axis in encoder shape {tuple(weights.shape)}")


def _factor_bases(cfg: FhmLayerSelfPatchingConfig, d_model: int, device: str) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    checkpoint = Path(cfg.checkpoint_dir)
    state = torch.load(checkpoint / "model_checkpoint.pt", map_location="cpu", weights_only=True)
    feature_indices = json.loads((checkpoint / "bilinear_feature_indices.json").read_text())["feature_indices"]
    image_factors = state["bil_U"].float()
    prompt_factors = state["bil_V"].float()
    assert image_factors.shape == prompt_factors.shape
    crosscoder = GemmaCrosscoder.load_from_hub(
        cfg.crosscoder_repo_id, cfg.crosscoder_hookpoint, n_layers=1, d_model=d_model, device=device,
    )
    feature_vectors = _encoder_vectors(crosscoder, feature_indices)
    assert feature_vectors.shape[1] == image_factors.shape[0]
    rank = image_factors.shape[1]
    image_basis = _orthonormal_columns(feature_vectors @ image_factors.to(device), rank)
    prompt_basis = _orthonormal_columns(feature_vectors @ prompt_factors.to(device), rank)
    return image_basis, prompt_basis, [int(index) for index in feature_indices]


def _random_basis(d_model: int, rank: int, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return _orthonormal_columns(torch.randn(d_model, rank, generator=generator, device=device), rank)


def _low_weight_basis(
    crosscoder: GemmaCrosscoder,
    excluded: list[int],
    calibration: torch.Tensor,
    reference_basis: torch.Tensor,
    rank: int,
    n_candidates: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    generator = torch.Generator(device=calibration.device).manual_seed(seed)
    candidates = torch.randperm(crosscoder.num_latents, generator=generator, device=calibration.device).tolist()
    excluded_set = set(excluded)
    candidates = [index for index in candidates if index not in excluded_set][:n_candidates]
    vectors = _encoder_vectors(crosscoder, candidates).to(calibration.device)
    vectors = vectors / vectors.norm(dim=0, keepdim=True).clamp_min(1e-8)
    energy = (calibration.float() @ vectors).square().mean(dim=0)
    reference_energy = (calibration.float() @ reference_basis).square().mean(dim=0)
    available = set(range(len(candidates)))
    selected_indices: list[int] = []
    for target_energy in reference_energy.tolist():
        selected_index = min(available, key=lambda index: abs(float(energy[index]) - target_energy))
        selected_indices.append(selected_index)
        available.remove(selected_index)
    selected = torch.tensor(selected_indices, device=calibration.device)
    basis = _orthonormal_columns(vectors[:, selected], rank)
    return basis, {
        "n_candidates": len(candidates),
        "n_selected": rank,
        "selected_mean_projection_energy": float(energy[selected].mean().item()),
        "reference_mean_projection_energy": float(reference_energy.mean().item()),
        "candidate_mean_projection_energy": float(energy.mean().item()),
    }


def _project(hidden: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (hidden.float() @ basis) @ basis.T


def _patch_role(
    target: torch.Tensor,
    source: torch.Tensor,
    target_positions: torch.Tensor,
    source_positions: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    source_values = _resample_positions(source, source_positions, len(target_positions))
    target_values = target[target_positions]
    target_projection = _project(target_values, basis)
    target_error = target_values.float() - target_projection
    updated_values = _project(source_values, basis) + target_error
    complement_delta = updated_values - _project(updated_values, basis) - target_error
    complement_scale = target_error.abs().max().clamp_min(1.0)
    diagnostics = {
        "absolute": float(complement_delta.abs().max().item()),
        "relative": float((complement_delta.abs().max() / complement_scale).item()),
    }
    patched = target.clone()
    patched[target_positions] = updated_values.to(target.dtype)
    return patched, diagnostics


def _make_patch_hook(
    source_hidden: torch.Tensor,
    source_roles: dict[str, torch.Tensor],
    target_roles: dict[str, torch.Tensor],
    image_basis: torch.Tensor | None,
    prompt_basis: torch.Tensor | None,
    *,
    wrong_role: bool = False,
    capture_residual: dict[str, float] | None = None,
):
    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden = _output_tensor(output)
        assert hidden.shape[0] == 1, "This compact experiment expects batch size one."
        patched = hidden[0]
        max_error = {"absolute": 0.0, "relative": 0.0}
        if image_basis is not None:
            target_role = "prompt" if wrong_role else "image"
            patched, diagnostics = _patch_role(
                patched, source_hidden, target_roles[target_role], source_roles["image"], image_basis,
            )
            max_error = {
                key: max(max_error[key], diagnostics[key]) for key in max_error
            }
        if prompt_basis is not None:
            target_role = "image" if wrong_role else "prompt"
            patched, diagnostics = _patch_role(
                patched, source_hidden, target_roles[target_role], source_roles["prompt"], prompt_basis,
            )
            max_error = {
                key: max(max_error[key], diagnostics[key]) for key in max_error
            }
        if capture_residual is not None:
            capture_residual.update({f"max_complement_error_{key}": value for key, value in max_error.items()})
        return _replace_output(output, patched.unsqueeze(0))
    return hook


def _make_source_self_patch_hook(
    roles: dict[str, torch.Tensor], image_basis: torch.Tensor, prompt_basis: torch.Tensor,
):
    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden = _output_tensor(output)
        _, image_diagnostics = _patch_role(
            hidden[0], hidden[0], roles["image"], roles["image"], image_basis,
        )
        _, prompt_diagnostics = _patch_role(
            hidden[0], hidden[0], roles["prompt"], roles["prompt"], prompt_basis,
        )
        logger.debug(
            "Source self-patch numerical reconstruction error: image=%s prompt=%s",
            image_diagnostics, prompt_diagnostics,
        )
        # P(x) + (I - P)(x) is mathematically x. Return x exactly so bfloat16
        # rounding and float32 cancellation cannot turn the identity control
        # into a spurious intervention.
        return output
    return hook


def _margin_from_outputs(outputs: Any, yes_ids: list[int], no_ids: list[int]) -> float:
    logits = outputs.logits[0, -1].float()
    return max(float(logits[index]) for index in yes_ids) - max(float(logits[index]) for index in no_ids)


def _forward(
    model: Any,
    inputs: dict[str, Any],
    yes_ids: list[int],
    no_ids: list[int],
    *,
    capture_module: nn.Module | None = None,
    patch_module: nn.Module | None = None,
    patch_hook: Any = None,
) -> tuple[float, torch.Tensor | None]:
    captured: list[torch.Tensor] = []
    handles = []
    if capture_module is not None:
        def capture(_module: nn.Module, _inputs: Any, output: Any) -> None:
            captured.append(_output_tensor(output)[0].detach().float())
        handles.append(capture_module.register_forward_hook(capture))
    if patch_module is not None:
        assert patch_hook is not None
        handles.append(patch_module.register_forward_hook(patch_hook))
    try:
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return _margin_from_outputs(outputs, yes_ids, no_ids), (captured[-1] if captured else None)


def _prompt(sample: MemeSample, mode: str) -> str:
    if mode == "harmfulness":
        return build_simple_classify_prompt(sample)
    assert mode == "neutral", f"Unknown prompt mode: {mode}"
    text = sample.meme_text or "[no visible text provided]"
    return (
        f'Describe the image and transcribe any visible text neutrally.\n'
        f'OCR text supplied for transcription checking: "{text}"\n'
        "Do not assess hatefulness, offensiveness, intent, or morality."
    )


def _inputs(processor: Any, sample: MemeSample, prompt_mode: str, device: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    with Image.open(sample.image_path) as image_file:
        image = image_file.convert("RGB")
    inputs = build_multimodal_inputs(processor, image, _prompt(sample, prompt_mode), device)
    return inputs, _role_positions(inputs["input_ids"][0], processor)


def _pair_rates(rows: list[dict[str, Any]], metadata_path: str) -> dict[str, Any]:
    if not metadata_path:
        return {}
    metadata = _load_fhm_confounder_metadata(Path(metadata_path))
    row_by_id = {str(row["sample_id"]): row for row in rows}
    result: dict[str, Any] = {}
    for source, column in {"pseudo_img": "pseudo_img_idx", "pseudo_text": "pseudo_text_idx"}.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample_id, row in row_by_id.items():
            group = _normalise_group_value(metadata.get(_normalise_fhm_id(sample_id), {}).get(column))
            if group is not None:
                groups[group].append(row)
        result[source] = {}
        for condition in rows[0]["margins"]:
            correct = 0
            n_pairs = 0
            for group_rows in groups.values():
                positive = [row for row in group_rows if row["gold"] == "hateful"]
                negative = [row for row in group_rows if row["gold"] == "not hateful"]
                for pos in positive:
                    for neg in negative:
                        n_pairs += 1
                        correct += pos["margins"][condition] > neg["margins"][condition]
            result[source][condition] = {
                "n_pairs": n_pairs,
                "positive_margin_rate": round(correct / n_pairs, 4) if n_pairs else None,
            }
    return result


def run(cfg: FhmLayerSelfPatchingConfig) -> None:
    assert cfg.checkpoint_dir, "--checkpoint_dir is required."
    device = resolve_visible_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = [sample for sample in load_meme_samples(cfg.dataset) if sample.task_name == "hateful_memes_binary"]
    assert samples, "No labeled FHM samples found."
    samples = sorted(samples, key=lambda sample: sample.sample_id)
    random.Random(cfg.seed).shuffle(samples)
    samples = samples[:cfg.max_samples]
    assert len(samples) >= 2, "Need at least two samples for the unrelated-donor control."
    assert all(sample.gold_fields.get("label") in {"hateful", "not hateful"} for sample in samples)

    model, processor = load_gemma3_model(cfg.gemma)
    source_module = model.get_submodule(cfg.source_module_path)
    target_module = model.get_submodule(cfg.target_module_path)
    d_model = int(model.config.text_config.hidden_size)
    image_basis, prompt_basis, top_features = _factor_bases(cfg, d_model, device)
    rank = image_basis.shape[1]
    random_basis = _random_basis(d_model, rank, cfg.seed + 1, device)
    logger.info("Loaded %d FHM samples; patching rank-%d factor subspaces from %s to %s", len(samples), rank, cfg.source_module_path, cfg.target_module_path)

    yes_ids, no_ids = get_yes_no_token_ids(processor)
    calibration_hidden: list[torch.Tensor] = []
    cached: list[tuple[MemeSample, dict[str, Any], dict[str, torch.Tensor], torch.Tensor, float]] = []
    for sample in samples:
        harmful_inputs, harmful_roles = _inputs(processor, sample, "harmfulness", device)
        margin, source_hidden = _forward(
            model, harmful_inputs, yes_ids, no_ids, capture_module=source_module,
        )
        assert source_hidden is not None
        cached.append((sample, harmful_inputs, harmful_roles, source_hidden, margin))
        if len(calibration_hidden) < cfg.low_weight_calibration_samples:
            calibration_hidden.append(torch.cat([
                source_hidden[harmful_roles["image"]], source_hidden[harmful_roles["prompt"]],
            ]))
    calibration = torch.cat(calibration_hidden)
    crosscoder = GemmaCrosscoder.load_from_hub(
        cfg.crosscoder_repo_id, cfg.crosscoder_hookpoint, n_layers=1, d_model=d_model, device=device,
    )
    low_basis, low_stats = _low_weight_basis(
        crosscoder, top_features, calibration, image_basis, rank, cfg.low_weight_candidate_features, cfg.seed + 2,
    )
    del crosscoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    conditions = (
        "baseline",
        "source_self_identity",
        "image_factor",
        "prompt_factor",
        "joint_factor_reconstruction_error_preserving",
        "random_subspace_norm_matched",
        "low_probe_weight_activity_matched",
        "unrelated_donor",
        "wrong_token_role",
        "neutral_visual_state_frozen",
    )
    for index, (sample, harmful_inputs, harmful_roles, source_hidden, baseline_margin) in enumerate(cached):
        donor = cached[(index + 1) % len(cached)]
        donor_hidden = donor[3]
        residual_checks: dict[str, float] = {}
        margins = {"baseline": baseline_margin}
        identity_margin, _ = _forward(
            model, harmful_inputs, yes_ids, no_ids,
            patch_module=source_module,
            patch_hook=_make_source_self_patch_hook(harmful_roles, image_basis, prompt_basis),
        )
        margins["source_self_identity"] = identity_margin

        def patched_margin(
            condition: str,
            source: torch.Tensor,
            image_q: torch.Tensor | None,
            prompt_q: torch.Tensor | None,
            *, wrong_role: bool = False,
            source_roles: dict[str, torch.Tensor] | None = None,
        ) -> None:
            check: dict[str, float] = {}
            hook = _make_patch_hook(
                source, source_roles or harmful_roles, harmful_roles, image_q, prompt_q,
                wrong_role=wrong_role, capture_residual=check,
            )
            margin, _ = _forward(
                model, harmful_inputs, yes_ids, no_ids, patch_module=target_module, patch_hook=hook,
            )
            margins[condition] = margin
            residual_checks[condition] = check["max_complement_error_relative"]

        patched_margin("image_factor", source_hidden, image_basis, None)
        patched_margin("prompt_factor", source_hidden, None, prompt_basis)
        patched_margin("joint_factor_reconstruction_error_preserving", source_hidden, image_basis, prompt_basis)
        patched_margin("random_subspace_norm_matched", source_hidden, random_basis, random_basis)
        patched_margin("low_probe_weight_activity_matched", source_hidden, low_basis, low_basis)
        patched_margin("unrelated_donor", donor_hidden, image_basis, prompt_basis, source_roles=donor[2])
        patched_margin("wrong_token_role", source_hidden, image_basis, prompt_basis, wrong_role=True)

        neutral_inputs, neutral_roles = _inputs(processor, sample, "neutral", device)
        _, neutral_hidden = _forward(model, neutral_inputs, yes_ids, no_ids, capture_module=source_module)
        assert neutral_hidden is not None
        neutral_image = _project(neutral_hidden[neutral_roles["image"]], image_basis)
        harmful_image = _project(source_hidden[harmful_roles["image"]], image_basis)
        projection_delta = float((harmful_image - neutral_image).norm(dim=-1).mean().item())
        harmful_norm = float(harmful_image.norm(dim=-1).mean().item())
        patched_margin(
            "neutral_visual_state_frozen", neutral_hidden, image_basis, None, source_roles=neutral_roles,
        )
        rows.append({
            "sample_id": sample.sample_id,
            "gold": sample.gold_fields["label"],
            "baseline_prediction": "hateful" if baseline_margin > 0 else "not hateful",
            "margins": {key: round(float(margins[key]), 6) for key in conditions},
            "margin_delta_from_baseline": {key: round(float(margins[key] - baseline_margin), 6) for key in conditions},
            "joint_superadditivity": round(float(
                margins["joint_factor_reconstruction_error_preserving"]
                - margins["image_factor"]
                - margins["prompt_factor"]
                + baseline_margin
            ), 6),
            "max_preserved_complement_relative_error": {key: round(value, 8) for key, value in residual_checks.items()},
            "harmful_minus_neutral_image_factor_l2": round(projection_delta, 6),
            "harmful_image_factor_l2": round(harmful_norm, 6),
        })
        if (index + 1) % 8 == 0 or index + 1 == len(cached):
            logger.info("Processed %d/%d samples", index + 1, len(cached))

    summary: dict[str, Any] = {
        "config": {
            "source_module_path": cfg.source_module_path,
            "target_module_path": cfg.target_module_path,
            "checkpoint_dir": cfg.checkpoint_dir,
            "n_factor_features": len(top_features),
            "factor_rank": rank,
            "seed": cfg.seed,
        },
        "controls": {
            "source_self_identity": "Apply P(source) + (I-P)(source) at layer 31; should produce zero margin delta.",
            "random_subspace_norm_matched": "Random orthonormal rank-r subspace with the same patch rank.",
            "low_probe_weight_activity_matched": "Encoder features outside the top bilinear set, selected by calibration projection energy.",
            "unrelated_donor": "Next deterministic unrelated FHM sample supplies role-aligned source states.",
            "wrong_token_role": "Image factor patched to prompt positions and prompt factor patched to image positions.",
            "joint_factor_reconstruction_error_preserving": "P(source) + (I-P)(target), preserving the target residual outside P; relative numerical preservation error is reported.",
        },
        "low_weight_selection": low_stats,
        "conditions": {},
        "visual_recoding": {},
    }
    for condition in conditions:
        margins = torch.tensor([row["margins"][condition] for row in rows])
        gold = [row["gold"] for row in rows]
        preds = ["hateful" if margin > 0 else "not hateful" for margin in margins.tolist()]
        deltas = torch.tensor([row["margin_delta_from_baseline"][condition] for row in rows])
        summary["conditions"][condition] = {
            "metrics": _per_class_metrics(gold, preds),
            "mean_margin": round(float(margins.mean()), 6),
            "mean_margin_delta": round(float(deltas.mean()), 6),
            "mean_abs_margin_delta": round(float(deltas.abs().mean()), 6),
            "n_prediction_changes": sum(pred != row["baseline_prediction"] for pred, row in zip(preds, rows)),
            "max_complement_relative_error": max((max(row["max_preserved_complement_relative_error"].values(), default=0.0) for row in rows), default=0.0),
        }
    recoding_delta = torch.tensor([row["harmful_minus_neutral_image_factor_l2"] for row in rows])
    recoding_norm = torch.tensor([row["harmful_image_factor_l2"] for row in rows])
    summary["visual_recoding"] = {
        "mean_harmful_minus_neutral_image_factor_l2": round(float(recoding_delta.mean()), 6),
        "mean_relative_change": round(float((recoding_delta / recoding_norm.clamp_min(1e-8)).mean()), 6),
        "neutral_visual_state_frozen": summary["conditions"]["neutral_visual_state_frozen"],
    }
    superadditivity = torch.tensor([row["joint_superadditivity"] for row in rows])
    summary["superadditivity"] = {
        "definition": "joint delta minus image-only delta minus prompt-only delta",
        "mean_margin_superadditivity": round(float(superadditivity.mean()), 6),
        "mean_abs_margin_superadditivity": round(float(superadditivity.abs().mean()), 6),
        "fraction_positive": round(float((superadditivity > 0).float().mean()), 6),
    }
    summary["fhm_pair_ordering"] = _pair_rates(rows, cfg.fhm_confounder_metadata_path)
    with (out_dir / "per_sample.jsonl").open("w") as fd:
        for row in rows:
            fd.write(json.dumps(row) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Wrote %s", out_dir / "summary.json")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse(FhmLayerSelfPatchingConfig, argument_generation_mode=ArgumentGenerationMode.BOTH))


if __name__ == "__main__":
    main()