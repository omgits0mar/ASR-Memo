"""SpeechTranscriber protocol (task T009).

Constitution II (real-time/streaming), V (hardware-aware), VI (per-turn language),
VII. Backend: ``nemotron_onnx`` (Nemotron 3.5 ASR Streaming, FP16 ONNX).

Cache-aware streaming: audio is fed as a continuous stream so the FastConformer
encoder/conv cache stays valid — the transcriber is NOT given pre-cut per-speaker
segments (would reset the cache; see plan.md Complexity Tracking).
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from ..types import AsrToken, AudioFrame, ComputeBackend

__all__ = ["SpeechTranscriber"]


@runtime_checkable
class SpeechTranscriber(Protocol):
    """Streaming, multilingual speech transcription (FR-005, FR-007, FR-008)."""

    def load(self, backend: ComputeBackend, *, precision: str = "fp16") -> None:
        """Load the ASR model onto the resolved backend. ``precision`` defaults to
        'fp16'; 'int8'/'int4' are opt-in (quantization gate, research Decision 8)."""
        ...

    def reset(self) -> None:
        """Clear the cache-aware streaming state for a new session."""
        ...

    def push(self, frame: AudioFrame, *, language_hint: str | None = None) -> List[AsrToken]:
        """Feed a 16 kHz mono chunk; return newly decoded tokens with timestamps,
        per-token detected language, and scores. Each frame processed exactly once
        (FR-016). ``language_hint`` biases recognition while still emitting per-token
        language."""
        ...

    def flush(self) -> List[AsrToken]:
        """Emit any remaining tokens at end-of-stream/stop."""
        ...

    def supported_languages(self) -> List[str]:
        """~40 language-locales."""
        ...
