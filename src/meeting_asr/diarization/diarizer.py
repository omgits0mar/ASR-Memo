"""SpeakerDiarizer protocol (task T008).

Constitution IV (first-class diarization, stable session IDs), V (hardware-aware),
VII. Backend: ``sortformer_coreml`` (Streaming Sortformer).

The diarizer is authoritative for *who spoke* and runs first/continuously on the
captured audio. It is advisory to ``fusion.aligner`` — it does NOT receive or
re-segment ASR input (see plan.md Complexity Tracking: parallel topology).
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from ..types import AudioFrame, ComputeBackend, DiarFrame

__all__ = ["SpeakerDiarizer"]


@runtime_checkable
class SpeakerDiarizer(Protocol):
    """Streaming speaker diarization (FR-003, FR-004)."""

    def load(self, backend: ComputeBackend) -> None:
        """Load the diarization model onto the resolved backend (CoreML → MPS/CPU)."""
        ...

    def reset(self) -> None:
        """Clear the arrival-order speaker cache (AOSC) for a new session.
        Labels restart at 'Speaker 1'."""
        ...

    def push(self, frame: AudioFrame) -> List[DiarFrame]:
        """Feed a 16 kHz mono chunk; return newly-resolved frame-level diarization
        decisions (~80 ms granularity) with stable speaker labels. Maintains
        internal state across calls."""
        ...

    def max_speakers(self) -> int:
        """Concurrent-speaker capacity (4 for Sortformer 4spk)."""
        ...
