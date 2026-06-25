"""Inference backend factory.

Keeps platform/model selection in one place so the public facade does not need
to know which concrete transcriber/diarizer pair belongs to a compute backend.
"""

from __future__ import annotations

from ..asr.nemotron_onnx import NemotronOnnxTranscriber
from ..asr.transcriber import SpeechTranscriber
from ..diarization.diarizer import SpeakerDiarizer
from ..diarization.sortformer_coreml import SortformerCoreMLDiarizer
from ..types import ComputeBackend


def build_inference_backends(backend: ComputeBackend) -> tuple[SpeakerDiarizer, SpeechTranscriber]:
    """Build the diarizer/transcriber pair for ``backend``."""
    if backend in (ComputeBackend.CUDA, ComputeBackend.TORCH_CPU):
        from ..asr.nemotron_nemo import NemotronNeMoTranscriber
        from ..diarization.sortformer_nemo import SortformerNeMoDiarizer

        return SortformerNeMoDiarizer(), NemotronNeMoTranscriber()
    return SortformerCoreMLDiarizer(), NemotronOnnxTranscriber()


__all__ = ["build_inference_backends"]
