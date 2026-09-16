from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ResidualStats:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-5

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device, x.dtype) + self.mean.to(x.device, x.dtype)


def save_residual_stats(path: Path, stats: ResidualStats) -> None:
    torch.save(
        {"mean": stats.mean.cpu(), "std": stats.std.cpu(), "eps": stats.eps},
        path / "residual_stats.pt",
    )


def load_residual_stats(path: Path, device: str = "cpu") -> ResidualStats:
    data = torch.load(path / "residual_stats.pt", map_location=device, weights_only=True)
    return ResidualStats(mean=data["mean"], std=data["std"], eps=data["eps"])