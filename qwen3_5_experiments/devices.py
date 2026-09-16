import os

import torch


def resolve_visible_device(requested: str) -> str:
    """Resolve the requested device string under CUDA_VISIBLE_DEVICES constraints."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd.strip():
        return requested
    if requested.startswith("cuda:"):
        requested_idx = int(requested.split(":")[1])
        visible = [int(x.strip()) for x in cvd.split(",") if x.strip()]
        if requested_idx < len(visible):
            return f"cuda:{requested_idx}"
    return requested