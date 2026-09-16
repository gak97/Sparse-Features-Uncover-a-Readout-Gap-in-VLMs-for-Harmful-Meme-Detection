import json
from dataclasses import dataclass
from pathlib import Path

import torch


STATS_FNAME = "residual_stats.json"


@dataclass(frozen=True)
class ResidualStats:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-5

    def __post_init__(self) -> None:
        assert self.mean.ndim == 1, f"Expected 1D mean, got shape={tuple(self.mean.shape)}"
        assert self.std.ndim == 1, f"Expected 1D std, got shape={tuple(self.std.shape)}"
        assert self.mean.shape == self.std.shape, (
            f"Mean/std shape mismatch: {tuple(self.mean.shape)} != {tuple(self.std.shape)}"
        )
        assert torch.isfinite(self.mean).all(), "ResidualStats.mean must be finite"
        assert torch.isfinite(self.std).all(), "ResidualStats.std must be finite"
        assert torch.all(self.std > 0), "ResidualStats.std must be positive"

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def to_dict(self) -> dict[str, object]:
        return {
            "eps": self.eps,
            "mean": self.mean.cpu().tolist(),
            "std": self.std.cpu().tolist(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "ResidualStats":
        mean = torch.tensor(raw["mean"], dtype=torch.float32)
        std = torch.tensor(raw["std"], dtype=torch.float32)
        eps = float(raw.get("eps", 1e-5))
        return cls(mean=mean, std=std, eps=eps)


def save_residual_stats(path: Path | str, stats: ResidualStats) -> None:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / STATS_FNAME).open("w") as fd:
        json.dump(stats.to_dict(), fd)


def load_residual_stats(path: Path | str) -> ResidualStats | None:
    candidate = Path(path)
    search_dirs = [candidate]
    if candidate.parent != candidate:
        search_dirs.append(candidate.parent)
    for directory in search_dirs:
        stats_fpath = directory / STATS_FNAME
        if not stats_fpath.exists():
            continue
        with stats_fpath.open() as fd:
            return ResidualStats.from_dict(json.load(fd))
    return None
