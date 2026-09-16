from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from gemma3_experiments.vendor_sae.sae import Sae

from .gemma import GemmaSae
from .normalization import ResidualStats


@dataclass
class JointSaeOutput:
    base_latents: torch.Tensor
    base_reconstruction: torch.Tensor
    residual_error: torch.Tensor
    residual_top_acts: torch.Tensor
    residual_top_indices: torch.Tensor | None
    residual_reconstruction: torch.Tensor
    combined_reconstruction: torch.Tensor


def run_joint_saes(
    hidden: torch.Tensor,
    base_sae: GemmaSae,
    residual_sae: Sae,
    residual_stats: ResidualStats | None = None,
) -> JointSaeOutput:
    base_latents, base_reconstruction = base_sae.reconstruct(hidden)
    residual_error = hidden - base_reconstruction
    residual_sae_in = residual_stats.normalize(residual_error) if residual_stats is not None else residual_error
    residual_top_acts, residual_top_indices = residual_sae.encode(residual_sae_in)
    residual_reconstruction = residual_sae.decode(residual_top_acts, residual_top_indices)
    if residual_stats is not None:
        residual_reconstruction = residual_stats.denormalize(residual_reconstruction)
    return JointSaeOutput(
        base_latents=base_latents,
        base_reconstruction=base_reconstruction,
        residual_error=residual_error,
        residual_top_acts=residual_top_acts,
        residual_top_indices=residual_top_indices,
        residual_reconstruction=residual_reconstruction,
        combined_reconstruction=base_reconstruction + residual_reconstruction,
    )


def apply_residual_feature_delta(
    hidden: torch.Tensor,
    base_sae: GemmaSae,
    residual_sae: Sae,
    feature_idx: int,
    delta: float,
    residual_stats: ResidualStats | None = None,
) -> tuple[torch.Tensor, JointSaeOutput]:
    return apply_residual_feature_steering(
        hidden,
        base_sae,
        residual_sae,
        feature_indices=[feature_idx],
        feature_deltas=[delta],
        residual_stats=residual_stats,
    )


def apply_residual_feature_steering(
    hidden: torch.Tensor,
    base_sae: GemmaSae,
    residual_sae: Sae,
    *,
    feature_indices: Sequence[int],
    feature_deltas: Sequence[float] | None = None,
    feature_values: Sequence[float] | None = None,
    mode: str = "add",
    residual_stats: ResidualStats | None = None,
) -> tuple[torch.Tensor, JointSaeOutput]:
    assert feature_indices, "Expected at least one feature index for steering"
    assert mode in {"add", "clamp"}, f"Unsupported feature steering mode: {mode!r}"
    if mode == "add":
        assert feature_deltas is not None, "feature_deltas are required for add steering"
        assert len(feature_indices) == len(feature_deltas), "feature_indices and feature_deltas must have the same length"
    if mode == "clamp":
        assert feature_values is not None, "feature_values are required for clamp steering"
        assert len(feature_indices) == len(feature_values), "feature_indices and feature_values must have the same length"

    joint = run_joint_saes(hidden, base_sae, residual_sae, residual_stats)
    residual_sae_in = residual_stats.normalize(joint.residual_error) if residual_stats is not None else joint.residual_error
    feature_i = torch.tensor(list(feature_indices), dtype=torch.long, device=residual_sae_in.device)
    n_latents = getattr(residual_sae, "num_latents", None)
    assert n_latents is not None, "Residual SAE must expose num_latents for feature steering"
    assert bool(((0 <= feature_i) & (feature_i < n_latents)).all()), f"Feature index out of range for num_latents={n_latents}"

    if hasattr(residual_sae, "select_topk"):
        pre_acts = residual_sae.pre_acts(residual_sae_in)
        if mode == "add":
            deltas = torch.tensor(list(feature_deltas or ()), dtype=pre_acts.dtype, device=pre_acts.device)
            pre_acts[..., feature_i] = pre_acts[..., feature_i] + deltas
        else:
            values = torch.tensor(list(feature_values or ()), dtype=pre_acts.dtype, device=pre_acts.device)
            pre_acts[..., feature_i] = values
        steered_acts, steered_indices = residual_sae.select_topk(pre_acts)
        steered_residual = residual_sae.decode(steered_acts, steered_indices)
    else:
        acts, act_indices = residual_sae.encode(residual_sae_in)
        assert act_indices is None, "Dense feature steering expected JumpReLU-style encode output with indices=None"
        steered_acts = acts.clone()
        if mode == "add":
            deltas = torch.tensor(list(feature_deltas or ()), dtype=steered_acts.dtype, device=steered_acts.device)
            steered_acts[..., feature_i] = torch.clamp_min(steered_acts[..., feature_i] + deltas, 0.0)
        else:
            values = torch.tensor(list(feature_values or ()), dtype=steered_acts.dtype, device=steered_acts.device)
            steered_acts[..., feature_i] = torch.clamp_min(values, 0.0)
        steered_residual = residual_sae.decode(steered_acts, None)
    if residual_stats is not None:
        steered_residual = residual_stats.denormalize(steered_residual)
    delta_hidden = steered_residual - joint.residual_reconstruction
    return hidden + delta_hidden, joint


def make_residual_steering_hook(
    base_sae: GemmaSae,
    residual_sae: Sae,
    *,
    feature_idx: int,
    delta: float,
    token_mask: torch.Tensor | None = None,
    residual_stats: ResidualStats | None = None,
) -> Callable[[nn.Module, tuple, torch.Tensor], torch.Tensor]:
    def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor):
        hidden = output[0] if isinstance(output, tuple) else output
        updated = hidden
        if token_mask is None:
            updated_hidden, _ = apply_residual_feature_delta(
                hidden,
                base_sae,
                residual_sae,
                feature_idx,
                delta,
                residual_stats,
            )
            updated = updated_hidden
        else:
            selected_hidden, _ = apply_residual_feature_delta(
                hidden[:, token_mask, :],
                base_sae,
                residual_sae,
                feature_idx,
                delta,
                residual_stats,
            )
            updated = hidden.clone()
            updated[:, token_mask, :] = selected_hidden
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    return hook


def make_joint_reconstruction_hook(
    base_sae: GemmaSae,
    residual_sae: Sae,
    *,
    feature_idx: int | None = None,
    feature_indices: Sequence[int] | None = None,
    delta: float = 0.0,
    feature_deltas: Sequence[float] | None = None,
    feature_values: Sequence[float] | None = None,
    steering_mode: str = "add",
    reconstruction_alpha: float = 1.0,
    residual_stats: ResidualStats | None = None,
) -> Callable[[nn.Module, tuple, torch.Tensor], torch.Tensor]:
    assert reconstruction_alpha >= 0.0, "reconstruction_alpha must be non-negative"
    if feature_indices is None:
        feature_indices = [feature_idx] if feature_idx is not None else None
    if feature_deltas is None and feature_idx is not None:
        feature_deltas = [delta]

    def hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor):
        hidden = output[0] if isinstance(output, tuple) else output
        if reconstruction_alpha == 0.0:
            return output
        with torch.no_grad():
            hidden_f32 = hidden.detach().to(dtype=torch.float32)
            if feature_indices is None:
                target = run_joint_saes(hidden_f32, base_sae, residual_sae, residual_stats).combined_reconstruction
            else:
                target, _ = apply_residual_feature_steering(
                    hidden_f32,
                    base_sae,
                    residual_sae,
                    feature_indices=feature_indices,
                    feature_deltas=feature_deltas,
                    feature_values=feature_values,
                    mode=steering_mode,
                    residual_stats=residual_stats,
                )
            updated = hidden_f32 + reconstruction_alpha * (target - hidden_f32)
            updated = updated.to(dtype=hidden.dtype).contiguous().detach()
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    return hook
