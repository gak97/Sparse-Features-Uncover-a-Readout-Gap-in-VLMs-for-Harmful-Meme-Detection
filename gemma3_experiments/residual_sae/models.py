import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from gemma3_experiments.vendor_sae.sae import Sae


class ResidualSaeForwardOutput(NamedTuple):
    sae_out: Tensor
    latent_acts: Tensor
    latent_indices: Tensor | None
    fvu: Tensor
    auxk_loss: Tensor
    multi_topk_fvu: Tensor
    l0: Tensor
    l0_loss: Tensor


@dataclass
class JumpReluSaeConfig:
    expansion_factor: int = 8
    num_latents: int = 0
    normalize_decoder: bool = True
    threshold_init: float = 0.1
    threshold_bandwidth: float = 0.1
    l0_coefficient: float = 1e-4


class JumpReluResidualSae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: JumpReluSaeConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert cfg.threshold_init > 0.0, "threshold_init must be positive"
        assert cfg.threshold_bandwidth > 0.0, "threshold_bandwidth must be positive"
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
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

    def encode(self, x: Tensor) -> tuple[Tensor, None]:
        return self._jump_relu(self.pre_acts(x))[0], None

    def decode(self, acts: Tensor, _indices: Tensor | None = None) -> Tensor:
        return acts.to(self.dtype) @ self.W_dec + self.b_dec

    def _jump_relu(self, pre_acts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        threshold = self.threshold.to(pre_acts.dtype)
        hard_mask = (pre_acts > threshold).to(pre_acts.dtype)
        soft_mask = torch.sigmoid((pre_acts - threshold) / self.cfg.threshold_bandwidth)
        ste_mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        acts = torch.relu(pre_acts) * ste_mask
        return acts, hard_mask, soft_mask

    def forward(self, x: Tensor, dead_mask: Tensor | None = None) -> ResidualSaeForwardOutput:
        del dead_mask
        pre_acts = self.pre_acts(x)
        acts, hard_mask, soft_mask = self._jump_relu(pre_acts)
        sae_out = self.decode(acts)
        total_variance = (x - x.mean(0)).pow(2).sum()
        fvu = (sae_out - x).pow(2).sum() / total_variance
        return ResidualSaeForwardOutput(
            sae_out=sae_out,
            latent_acts=acts,
            latent_indices=None,
            fvu=fvu,
            auxk_loss=sae_out.new_tensor(0.0),
            multi_topk_fvu=sae_out.new_tensor(0.0),
            l0=hard_mask.sum(dim=-1).float().mean(),
            l0_loss=self.cfg.l0_coefficient * soft_mask.sum(dim=-1).mean(),
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
                    "architecture": "jump_relu",
                    "d_in": self.d_in,
                    **asdict(self.cfg),
                },
                fd,
            )

    @staticmethod
    def load_from_disk(path: Path | str, device: str | torch.device = "cpu") -> "JumpReluResidualSae":
        path = Path(path)
        with (path / "cfg.json").open() as fd:
            cfg_dict = json.load(fd)
        cfg_dict.pop("architecture", None)
        d_in = cfg_dict.pop("d_in")
        sae = JumpReluResidualSae(d_in, JumpReluSaeConfig(**cfg_dict), device=device)
        load_model(sae, str(path / "sae.safetensors"), device=str(device), strict=True)
        sae.eval()
        return sae


def load_residual_sae(path: Path | str, device: str | torch.device = "cpu"):
    path = Path(path)
    with (path / "cfg.json").open() as fd:
        cfg_dict = json.load(fd)
    if cfg_dict.get("architecture") == "jump_relu":
        return JumpReluResidualSae.load_from_disk(path, device=device)
    return Sae.load_from_disk(path, device=device)
