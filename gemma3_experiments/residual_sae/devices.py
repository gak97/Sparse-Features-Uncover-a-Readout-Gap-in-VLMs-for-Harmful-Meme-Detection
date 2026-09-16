import logging

import torch


logger = logging.getLogger(__name__)


def resolve_visible_device(requested_device: str) -> str:
    if not requested_device.startswith("cuda"):
        return requested_device
    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {requested_device}, but CUDA is not available")

    n_devices = torch.cuda.device_count()
    if n_devices <= 0:
        raise RuntimeError(f"Requested CUDA device {requested_device}, but no visible CUDA devices were found")

    if ":" not in requested_device:
        return "cuda:0"

    _, index_str = requested_device.split(":", maxsplit=1)
    try:
        requested_index = int(index_str)
    except ValueError as exc:
        raise RuntimeError(f"Unsupported CUDA device string: {requested_device}") from exc

    if 0 <= requested_index < n_devices:
        return requested_device

    fallback_device = "cuda:0"
    logger.warning(
        "Requested device %s is invalid for %d visible CUDA device(s); using %s instead",
        requested_device,
        n_devices,
        fallback_device,
    )
    return fallback_device
