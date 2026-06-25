"""Unit tests for backends/device.py (T006; research Decision 5)."""

from __future__ import annotations

from meeting_asr.backends.device import (
    DeviceProbe,
    compute_units_label,
    ort_providers,
    resolve_backend,
)
from meeting_asr.types import ComputeBackend


def probe(*, silicon, ort, coreml, torch=False, cuda=False, prefer_ane=False):
    return DeviceProbe(
        is_apple_silicon=lambda: silicon,
        has_onnxruntime=lambda: ort,
        has_coreml_ep=lambda: coreml,
        has_torch=lambda: torch,
        has_cuda=lambda: cuda,
        prefer_ane=prefer_ane,
    )


class TestResolveBackend:
    def test_default_apple_silicon_picks_gpu_cpu(self):
        b = resolve_backend(probe(silicon=True, ort=True, coreml=True))
        assert b is ComputeBackend.COREML_GPU_CPU

    def test_opt_in_ane_profile(self):
        b = resolve_backend(probe(silicon=True, ort=True, coreml=True, prefer_ane=True))
        assert b is ComputeBackend.COREML_ANE

    def test_no_coreml_falls_back_to_cpu(self):
        b = resolve_backend(probe(silicon=True, ort=True, coreml=False))
        assert b is ComputeBackend.CPU

    def test_non_apple_silicon_forces_cpu_even_if_coreml_listed(self):
        b = resolve_backend(probe(silicon=False, ort=True, coreml=True))
        assert b is ComputeBackend.CPU

    def test_cuda_wins_after_apple_coreml_path(self):
        b = resolve_backend(probe(silicon=False, ort=True, coreml=False, torch=True, cuda=True))
        assert b is ComputeBackend.CUDA

    def test_torch_cpu_wins_before_onnx_cpu(self):
        b = resolve_backend(probe(silicon=False, ort=True, coreml=False, torch=True, cuda=False))
        assert b is ComputeBackend.TORCH_CPU

    def test_nothing_installed_still_cpu(self):
        assert resolve_backend(probe(silicon=False, ort=False, coreml=False)) is ComputeBackend.CPU


class TestProviderConfig:
    def test_coreml_gpu_cpu_pairs_with_cpu_fallback(self):
        provs = ort_providers(ComputeBackend.COREML_GPU_CPU)
        names = [p[0] if isinstance(p, tuple) else p for p in provs]
        assert names == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        # MLComputeUnits must be CPUAndGPU for the default
        coreml_opts = provs[0][1]
        assert coreml_opts["MLComputeUnits"] == "CPUAndGPU"

    def test_ane_profile_uses_all_compute_units(self):
        provs = ort_providers(ComputeBackend.COREML_ANE)
        assert provs[0][1]["MLComputeUnits"] == "All"

    def test_cpu_backend_uses_only_cpu_ep(self):
        assert ort_providers(ComputeBackend.CPU) == ["CPUExecutionProvider"]

    def test_compute_units_label(self):
        assert compute_units_label(ComputeBackend.COREML_GPU_CPU) == "CPUAndGPU"
        assert compute_units_label(ComputeBackend.COREML_ANE) == "All"
        assert compute_units_label(ComputeBackend.CPU) == "cpu"
        assert compute_units_label(ComputeBackend.CUDA) == "cuda"
        assert compute_units_label(ComputeBackend.TORCH_CPU) == "torch-cpu"
