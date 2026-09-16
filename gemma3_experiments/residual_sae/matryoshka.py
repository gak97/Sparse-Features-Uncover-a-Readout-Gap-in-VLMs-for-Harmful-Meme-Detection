import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors.torch import load_model, save_model
from torch import Tensor, nn


class MatryoshkaForwardOutput(NamedTuple):
    sae_out: Tensor
    latent_acts: Tensor
    hard_mask: Tensor
    loss: Tensor
    fvu: Tensor
    prefix_fvu: dict[int, Tensor]
    l0: Tensor
    l0_loss: Tensor


@dataclass
class MatryoshkaJumpReluConfig:
    """JumpReLU SAE with nested prefix reconstruction losses.

    `prefix_sizes` defines the Matryoshka levels. During training, each prefix
    reconstructs the same input using only the first k latents, which pressures
    early latents to carry broad/high-variance structure and later latents to add
    finer residual structure.
    """

    expansion_factor: int = 8
    num_latents: int = 0
    prefix_sizes: list[int] = field(default_factory=lambda: [256, 1024, 4096, 16384])
    prefix_loss_weights: list[float] = field(default_factory=list)
    normalize_decoder: bool = True
    threshold_init: float = 0.1
    threshold_bandwidth: float = 0.1
    l0_coefficient: float = 3e-4


class MatryoshkaJumpReluSae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: MatryoshkaJumpReluConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert cfg.threshold_init > 0.0, "threshold_init must be positive"
        assert cfg.threshold_bandwidth > 0.0, "threshold_bandwidth must be positive"
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
        self.prefix_sizes = self._validate_prefix_sizes(cfg.prefix_sizes)
        self.prefix_loss_weights = self._resolve_prefix_weights(cfg.prefix_loss_weights)

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))
        self.log_threshold = nn.Parameter(
            torch.full(
                (self.num_latents,),
                torch.log(torch.tensor(cfg.threshold_init, dtype=dtype or torch.float32)).item(),
                dtype=dtype,
                device=device,
            )
        )
        if cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

    def _validate_prefix_sizes(self, raw: list[int]) -> list[int]:
        prefixes = sorted({int(k) for k in raw if 0 < int(k) <= self.num_latents})
        if not prefixes or prefixes[-1] != self.num_latents:
            prefixes.append(self.num_latents)
        assert prefixes[-1] <= self.num_latents
        assert all(a < b for a, b in zip(prefixes, prefixes[1:])), prefixes
        return prefixes

    def _resolve_prefix_weights(self, raw: list[float]) -> list[float]:
        if raw:
            assert len(raw) == len(self.prefix_sizes), (
                f"prefix_loss_weights length {len(raw)} must match prefix_sizes length {len(self.prefix_sizes)}"
            )
            weights = [float(w) for w in raw]
        else:
            weights = [1.0 / len(self.prefix_sizes)] * len(self.prefix_sizes)
        assert all(w >= 0 for w in weights), weights
        assert sum(weights) > 0, weights
        total = sum(weights)
        return [w / total for w in weights]

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    @property
    def threshold(self) -> Tensor:
        return self.log_threshold.exp()

    def pre_acts(self, x: Tensor) -> Tensor:
        return self.encoder(x.to(self.dtype) - self.b_dec)

    def _jump_relu(self, pre_acts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        threshold = self.threshold.to(pre_acts.dtype)
        hard_mask = (pre_acts > threshold).to(pre_acts.dtype)
        soft_mask = torch.sigmoid((pre_acts - threshold) / self.cfg.threshold_bandwidth)
        ste_mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        acts = torch.relu(pre_acts) * ste_mask
        return acts, hard_mask, soft_mask

    def encode(self, x: Tensor) -> tuple[Tensor, None]:
        return self._jump_relu(self.pre_acts(x))[0], None

    def decode(self, acts: Tensor, _indices: Tensor | None = None, *, prefix_size: int | None = None) -> Tensor:
        if prefix_size is None:
            prefix_size = acts.shape[-1]
        return acts[..., :prefix_size].to(self.dtype) @ self.W_dec[:prefix_size] + self.b_dec

    def forward(self, x: Tensor) -> MatryoshkaForwardOutput:
        pre_acts = self.pre_acts(x)
        acts, hard_mask, soft_mask = self._jump_relu(pre_acts)

        total_variance = (x - x.mean(0)).pow(2).sum().clamp_min(torch.finfo(x.dtype).eps)
        prefix_fvu: dict[int, Tensor] = {}
        recon_loss = x.new_tensor(0.0)
        sae_out = None
        for k, weight in zip(self.prefix_sizes, self.prefix_loss_weights):
            recon = self.decode(acts, prefix_size=k)
            fvu = (recon - x).pow(2).sum() / total_variance
            prefix_fvu[k] = fvu
            recon_loss = recon_loss + float(weight) * fvu
            if k == self.num_latents:
                sae_out = recon
        assert sae_out is not None

        l0 = hard_mask.sum(dim=-1).float().mean()
        l0_loss = self.cfg.l0_coefficient * soft_mask.sum(dim=-1).mean()
        return MatryoshkaForwardOutput(
            sae_out=sae_out,
            latent_acts=acts,
            hard_mask=hard_mask,
            loss=recon_loss + l0_loss,
            fvu=prefix_fvu[self.num_latents],
            prefix_fvu=prefix_fvu,
            l0=l0,
            l0_loss=l0_loss,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.W_dec.dtype).eps
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True) + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        if self.W_dec.grad is None:
            return
        parallel_component = (self.W_dec.grad * self.W_dec.data).sum(dim=1)
        self.W_dec.grad -= parallel_component[:, None] * self.W_dec.data

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_model(self, str(path / "sae.safetensors"))
        with (path / "cfg.json").open("w") as fd:
            json.dump(
                {
                    "architecture": "matryoshka_jump_relu",
                    "d_in": self.d_in,
                    **asdict(self.cfg),
                    "resolved_prefix_sizes": self.prefix_sizes,
                    "resolved_prefix_loss_weights": self.prefix_loss_weights,
                },
                fd,
                indent=2,
            )

    @staticmethod
    def load_from_disk(path: Path | str, device: str | torch.device = "cpu") -> "MatryoshkaJumpReluSae":
        path = Path(path)
        with (path / "cfg.json").open() as fd:
            cfg_dict = json.load(fd)
        cfg_dict.pop("architecture", None)
        cfg_dict.pop("resolved_prefix_sizes", None)
        cfg_dict.pop("resolved_prefix_loss_weights", None)
        d_in = cfg_dict.pop("d_in")
        sae = MatryoshkaJumpReluSae(d_in, MatryoshkaJumpReluConfig(**cfg_dict), device=device)
        load_model(sae, str(path / "sae.safetensors"), device=str(device), strict=True)
        sae.eval()
        return sae


def load_matryoshka_sae(path: Path | str, device: str | torch.device = "cpu") -> MatryoshkaJumpReluSae:
    path = Path(path)
    with (path / "cfg.json").open() as fd:
        cfg = json.load(fd)
    assert cfg.get("architecture") == "matryoshka_jump_relu", f"Not an MSAE checkpoint: {path}"
    return MatryoshkaJumpReluSae.load_from_disk(path, device=device)
