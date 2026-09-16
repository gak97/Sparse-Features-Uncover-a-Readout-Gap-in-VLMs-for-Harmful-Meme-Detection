

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .models import JumpReluResidualSae
from .qwen import QwenSae

try:
    from gemma3_experiments.residual_sae.normalization import ResidualStats
except ImportError:
    ResidualStats = None  # type: ignore[assignment, misc]


@dataclass
class JointSaeOutput:
    base_latents: torch.Tensor
    base_reconstruction: torch.Tensor
    residual_error: torch.Tensor
    residual_acts: torch.Tensor
    residual_reconstruction: torch.Tensor
    combined_reconstruction: torch.Tensor


def run_joint_saes(
    hidden: torch.Tensor,
    base_sae: QwenSae,
    residual_sae: JumpReluResidualSae,
    residual_stats=None,
) -> JointSaeOutput:
    """Run base SAE then residual SAE; return combined reconstruction."""
    base_latents, base_reconstruction = base_sae.reconstruct(hidden)
    residual_error = hidden - base_reconstruction
    sae_in = residual_stats.normalize(residual_error) if residual_stats is not None else residual_error
    residual_acts, _ = residual_sae.encode(sae_in)
    residual_recon = residual_sae.decode(residual_acts)
    if residual_stats is not None:
        residual_recon = residual_stats.denormalize(residual_recon)
    return JointSaeOutput(
        base_latents=base_latents,
        base_reconstruction=base_reconstruction,
        residual_error=residual_error,
        residual_acts=residual_acts,
        residual_reconstruction=residual_recon,
        combined_reconstruction=base_reconstruction + residual_recon,
    )


def apply_residual_feature_steering(
    hidden: torch.Tensor,
    base_sae: QwenSae,
    residual_sae: JumpReluResidualSae,
    *,
    feature_indices: Sequence[int],
    feature_deltas: Sequence[float] | None = None,
    feature_values: Sequence[float] | None = None,
    mode: str = "add",
    residual_stats=None,
) -> tuple[torch.Tensor, JointSaeOutput]:
    """Steer specific residual SAE features and return modified hidden state."""
    assert feature_indices, "Expected at least one feature index."
    assert mode in {"add", "clamp"}, f"Unknown mode: {mode!r}"
    joint = run_joint_saes(hidden, base_sae, residual_sae, residual_stats)

    sae_in = residual_stats.normalize(joint.residual_error) if residual_stats is not None else joint.residual_error
    acts, _ = residual_sae.encode(sae_in)
    steered_acts = acts.clone()

    feature_i = torch.tensor(list(feature_indices), dtype=torch.long, device=sae_in.device)
    if mode == "add":
        assert feature_deltas is not None
        deltas = torch.tensor(list(feature_deltas), dtype=steered_acts.dtype, device=sae_in.device)
        steered_acts[..., feature_i] = torch.clamp_min(steered_acts[..., feature_i] + deltas, 0.0)
    else:  # clamp
        assert feature_values is not None
        values = torch.tensor(list(feature_values), dtype=steered_acts.dtype, device=sae_in.device)
        steered_acts[..., feature_i] = torch.clamp_min(values, 0.0)

    steered_recon = residual_sae.decode(steered_acts)
    if residual_stats is not None:
        steered_recon = residual_stats.denormalize(steered_recon)
    delta_hidden = steered_recon - joint.residual_reconstruction
    return hidden + delta_hidden, joint


def make_joint_reconstruction_hook(
    base_sae: QwenSae,
    residual_sae: JumpReluResidualSae,
    *,
    feature_indices: Sequence[int] | None = None,
    feature_deltas: Sequence[float] | None = None,
    feature_values: Sequence[float] | None = None,
    steering_mode: str = "add",
    reconstruction_alpha: float = 1.0,
    residual_stats=None,
) -> Callable[[nn.Module, tuple, torch.Tensor], torch.Tensor]:
    pass
    assert reconstruction_alpha >= 0.0, "reconstruction_alpha must be non-negative"

    def hook(_module: nn.Module, _inputs: tuple, output) -> torch.Tensor:
        hidden = output[0] if isinstance(output, tuple) else output
        if reconstruction_alpha == 0.0:
            return output
        with torch.no_grad():
            h_f32 = hidden.detach().to(dtype=torch.float32)
            if feature_indices is not None:
                target, _ = apply_residual_feature_steering(
                    h_f32, base_sae, residual_sae,
                    feature_indices=feature_indices,
                    feature_deltas=feature_deltas,
                    feature_values=feature_values,
                    mode=steering_mode,
                    residual_stats=residual_stats,
                )
            else:
                target = run_joint_saes(h_f32, base_sae, residual_sae, residual_stats).combined_reconstruction
            updated = h_f32 + reconstruction_alpha * (target - h_f32)
            updated = updated.to(dtype=hidden.dtype).contiguous().detach()
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    return hook
