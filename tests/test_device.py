import sys
from types import SimpleNamespace

import pytest

from poomsae_scoring.device import describe_inference_device, resolve_inference_device


class FakeCuda:
    def __init__(self, available: bool, count: int = 0) -> None:
        self.available = available
        self.count = count

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def get_device_name(self, index: int) -> str:
        return f"Test GPU {index}"


def install_fake_torch(monkeypatch: pytest.MonkeyPatch, cuda: FakeCuda) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=cuda),
    )


def test_auto_selects_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_torch(monkeypatch, FakeCuda(available=True, count=1))

    assert resolve_inference_device("auto") == "cuda:0"


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_torch(monkeypatch, FakeCuda(available=False))

    assert resolve_inference_device("auto") == "cpu"


def test_explicit_cuda_fails_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_torch(monkeypatch, FakeCuda(available=False))

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_inference_device("cuda:0")


def test_out_of_range_cuda_device_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_torch(monkeypatch, FakeCuda(available=True, count=1))

    with pytest.raises(RuntimeError, match="only 1 device"):
        resolve_inference_device("cuda:1")


def test_describe_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_torch(monkeypatch, FakeCuda(available=True, count=1))

    assert describe_inference_device("cuda:0") == "Inference device: cuda:0 (Test GPU 0)"
