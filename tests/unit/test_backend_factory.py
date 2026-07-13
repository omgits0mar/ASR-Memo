from __future__ import annotations

from meeting_asr.asr.nemotron_nemo import NemotronNeMoTranscriber
from meeting_asr.asr.nemotron_onnx import NemotronOnnxTranscriber
from meeting_asr.backends.factory import build_inference_backends
from meeting_asr.diarization.sortformer_coreml import SortformerCoreMLDiarizer
from meeting_asr.diarization.sortformer_nemo import SortformerNeMoDiarizer
from meeting_asr.types import ComputeBackend


def test_factory_keeps_apple_backends_for_coreml():
    diarizer, transcriber = build_inference_backends(ComputeBackend.COREML_GPU_CPU)
    assert isinstance(diarizer, SortformerCoreMLDiarizer)
    assert isinstance(transcriber, NemotronOnnxTranscriber)


def test_factory_uses_nemo_backends_for_cuda():
    diarizer, transcriber = build_inference_backends(ComputeBackend.CUDA)
    assert isinstance(diarizer, SortformerNeMoDiarizer)
    assert isinstance(transcriber, NemotronNeMoTranscriber)


def test_factory_uses_nemo_backends_for_torch_cpu():
    diarizer, transcriber = build_inference_backends(ComputeBackend.TORCH_CPU)
    assert isinstance(diarizer, SortformerNeMoDiarizer)
    assert isinstance(transcriber, NemotronNeMoTranscriber)
