"""Hardware-aware backend resolver (task T006; research Decision 5).

Picks, at load time:
  1. Apple Silicon (default) — ONNX Runtime CoreML EP with
     ``MLComputeUnits = .cpuAndGPU`` (Metal GPU + CPU) for ASR; CoreML for
     diarization. ANE (``.all``) is an opt-in profile.
  2. CUDA-capable PyTorch/NeMo on non-Apple hosts.
  3. PyTorch/NeMo CPU when CUDA is absent.
  4. ONNX Runtime CPU EP fallback.

The resolver is pure logic over an injectable :class:`DeviceProbe`, so it is
fully unit-testable without onnxruntime/CoreML installed. The real probes do
lazy detection at call time.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Callable, Optional

from .._logging import get_logger
from ..types import ComputeBackend

_log = get_logger("backends.device")

# MLComputeUnits values for CoreMLExecutionProvider provider-options.
_COREML_COMPUTE_UNITS = {
    ComputeBackend.COREML_GPU_CPU: "CPUAndGPU",   # .cpuAndGPU → predictable FP16 on Metal
    ComputeBackend.COREML_ANE: "All",             # .all → ANE opt-in profile
}


@dataclass
class DeviceProbe:
    """Injectable hardware/runtime detection (so resolve_backend is testable)."""

    is_apple_silicon: Callable[[], bool]
    has_onnxruntime: Callable[[], bool]
    has_coreml_ep: Callable[[], bool]
    has_torch: Callable[[], bool]
    has_cuda: Callable[[], bool]
    prefer_ane: bool = False


def _is_apple_silicon() -> bool:
    if platform.system() != "Darwin":
        return False
    machine = platform.machine().lower()
    return machine == "arm64" or machine.startswith("aarch64") or machine.startswith("arm")


def _has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


def _has_coreml_ep() -> bool:
    """True iff onnxruntime is importable AND exposes the CoreML Execution Provider."""
    try:
        import onnxruntime as ort

        return "CoreMLExecutionProvider" in ort.get_available_providers()
    except Exception:  # pragma: no cover - environment dependent
        return False


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - environment dependent
        return False


def default_probe(*, prefer_ane: bool = False) -> DeviceProbe:
    """Real detection probes (lazy imports, called at resolve time)."""
    return DeviceProbe(
        is_apple_silicon=_is_apple_silicon,
        has_onnxruntime=_has_onnxruntime,
        has_coreml_ep=_has_coreml_ep,
        has_torch=_has_torch,
        has_cuda=_has_cuda,
        prefer_ane=prefer_ane,
    )


def resolve_backend(probe: Optional[DeviceProbe] = None) -> ComputeBackend:
    """Resolve the inference backend for this machine (Decision 5).

    Order: opt-in ANE → CoreML GPU+CPU → NeMo CUDA → NeMo torch-CPU → ONNX CPU.
    """
    p = probe or default_probe()
    is_apple = p.is_apple_silicon()
    has_coreml = p.has_coreml_ep()
    if p.prefer_ane and has_coreml and is_apple:
        _log.info("resolved CoreML EP with ANE (opt-in profile)")
        return ComputeBackend.COREML_ANE
    if has_coreml and is_apple:
        _log.info("resolved CoreML EP GPU+CPU (.cpuAndGPU) — default")
        return ComputeBackend.COREML_GPU_CPU
    if p.has_torch() and p.has_cuda():
        _log.info("resolved NeMo/PyTorch CUDA backend")
        return ComputeBackend.CUDA
    if p.has_torch():
        _log.info("resolved NeMo/PyTorch CPU backend")
        return ComputeBackend.TORCH_CPU
    if p.has_onnxruntime():
        _log.info("resolved ONNX Runtime CPU EP (fallback)")
        return ComputeBackend.CPU
    _log.info("resolved CPU fallback (no onnxruntime/CoreML)")
    return ComputeBackend.CPU


def compute_units_label(backend: ComputeBackend) -> str:
    """The CoreML ``MLComputeUnits`` value (or a human label for non-CoreML)."""
    return _COREML_COMPUTE_UNITS.get(backend, backend.value)


def ort_providers(backend: ComputeBackend) -> list:
    """The ONNX Runtime ``providers`` list for a backend (CoreML EP first, CPU last).

    CoreML is always paired with the CPU EP so unsupported ops fall back rather
    than error (FastConformer op coverage).
    """
    if backend is ComputeBackend.COREML_GPU_CPU:
        return [
            ("CoreMLExecutionProvider", {"MLComputeUnits": _COREML_COMPUTE_UNITS[backend]}),
            "CPUExecutionProvider",
        ]
    if backend is ComputeBackend.COREML_ANE:
        return [
            ("CoreMLExecutionProvider", {"MLComputeUnits": _COREML_COMPUTE_UNITS[backend]}),
            "CPUExecutionProvider",
        ]
    # CPU (and MPS handled out-of-band by the reference path).
    return ["CPUExecutionProvider"]


__all__ = [
    "DeviceProbe",
    "default_probe",
    "resolve_backend",
    "compute_units_label",
    "ort_providers",
]
