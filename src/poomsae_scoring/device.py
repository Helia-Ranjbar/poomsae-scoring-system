from __future__ import annotations

import importlib
import re
from types import ModuleType

CUDA_DEVICE_PATTERN = re.compile(r"cuda(?::(?P<index>\d+))?")


def _load_torch() -> ModuleType:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for YOLO inference. Install a CUDA-enabled "
            "PyTorch build before requesting a GPU."
        ) from error


def resolve_inference_device(requested: str = "auto") -> str:
    device = requested.strip().lower()
    if device == "cpu":
        return "cpu"
    if device == "auto":
        try:
            torch = _load_torch()
        except RuntimeError:
            return "cpu"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    match = CUDA_DEVICE_PATTERN.fullmatch(device)
    if match is None:
        raise ValueError("Device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'.")

    torch = _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access an NVIDIA GPU. "
            "Check the NVIDIA driver and install a CUDA-enabled PyTorch build."
        )

    device_index = int(match.group("index") or 0)
    device_count = torch.cuda.device_count()
    if device_index >= device_count:
        raise RuntimeError(
            f"CUDA device {device_index} was requested, but only {device_count} "
            "device(s) are available."
        )
    return f"cuda:{device_index}"


def describe_inference_device(device: str) -> str:
    if device == "cpu":
        return "Inference device: CPU"

    torch = _load_torch()
    device_index = int(device.split(":", maxsplit=1)[1])
    return f"Inference device: {device} ({torch.cuda.get_device_name(device_index)})"
