"""Core domain types for meeting_asr.

All entities are in-process Python ``dataclass`` objects (see
``specs/001-meeting-asr-backend/data-model.md``). Times are **seconds** (float)
relative to a single monotonic **session clock** that starts at session start.

This module is dependency-free (stdlib only) so every other module can import it
without pulling in heavy ML/runtime deps — that is what lets the whole pipeline
be exercised offline with fakes/fixtures (Constitution VII).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:  # numpy only needed for type hints, not at runtime → types stays dep-free
    import numpy as np


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class AudioSourceKind(str, Enum):
    """Local vs. meeting-app audio origin."""

    MICROPHONE = "microphone"
    SYSTEM = "system"


class SessionStatus(str, Enum):
    """TranscriptionSession state-machine states."""

    CREATED = "created"
    PREPARING = "preparing"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class CaptureState(str, Enum):
    """Live state of an audio source."""

    IDLE = "idle"
    CAPTURING = "capturing"
    PERMISSION_DENIED = "permission_denied"
    DEVICE_LOST = "device_lost"
    ERROR = "error"


class ModelKind(str, Enum):
    """Which pipeline stage a model serves."""

    ASR = "asr"
    DIARIZER = "diarizer"


class ModelFramework(str, Enum):
    """Runtime/model artifact format for a downloadable asset."""

    COREML = "coreml"
    ONNX = "onnx"
    NEMO = "nemo"


class ModelState(str, Enum):
    """Lifecycle of a downloadable model asset."""

    ABSENT = "absent"
    DOWNLOADING = "downloading"
    CACHED = "cached"
    LOADED = "loaded"
    ERROR = "error"


class ConfidenceBand(str, Enum):
    """Coarse quality bucket derived from a 0..1 score (FR-018)."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ComputeBackend(str, Enum):
    """Resolved inference backend + compute units (research Decision 5).

    The string value is what surfaces in ``SystemReadinessReport.compute_backend``.
    """

    COREML_GPU_CPU = "coreml-gpu+cpu"   # ONNX RT CoreML EP, MLComputeUnits=.cpuAndGPU (default)
    COREML_ANE = "coreml-ane"           # opt-in: MLComputeUnits=.all (ANE)
    MPS = "mps"                         # reference PyTorch-MPS path
    CPU = "cpu"                         # ONNX RT CPU EP fallback
    CUDA = "cuda"                       # NeMo/PyTorch CUDA path
    TORCH_CPU = "torch-cpu"             # NeMo/PyTorch CPU path


# Thresholds for score → band derivation (FR-018). Tunable but centralized here.
_BAND_THRESHOLDS = (
    (0.85, ConfidenceBand.HIGH),
    (0.60, ConfidenceBand.MEDIUM),
    (0.30, ConfidenceBand.LOW),
)


def band_for(score: float) -> ConfidenceBand:
    """Map a 0..1 confidence score to a coarse band (UNKNOWN for <0.30 or NaN)."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return ConfidenceBand.UNKNOWN
    if s != s or s < 0:  # NaN or negative → unknown
        return ConfidenceBand.UNKNOWN
    for threshold, band in _BAND_THRESHOLDS:
        if s >= threshold:
            return band
    return ConfidenceBand.UNKNOWN


def new_id(prefix: str = "") -> str:
    """Generate a stable uuid4 string, optionally namespaced by a prefix."""
    u = str(uuid.uuid4())
    return f"{prefix}{u}" if prefix else u


# --------------------------------------------------------------------------- #
# Supporting value types (frozen)
# --------------------------------------------------------------------------- #


@dataclass
class AudioFrame:
    """One chunk of captured audio, already at 16 kHz mono float32.

    ``pcm`` is a ``numpy.ndarray`` of dtype float32 (16 kHz mono). ``t_start``/
    ``t_end`` are session-clock seconds; gaps (silence/dropout) are preserved
    rather than collapsed (audio_capture contract). Mutable (pcm is a buffer).
    """

    pcm: "np.ndarray"
    t_start: float
    t_end: float
    source: AudioSourceKind

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def n_samples(self) -> int:
        return int(len(self.pcm))


@dataclass(frozen=True)
class DiarFrame:
    """A frame-level diarization decision (~80 ms granularity)."""

    t_start: float
    t_end: float
    speaker_label: str
    score: float = 1.0


@dataclass(frozen=True)
class AsrToken:
    """One decoded ASR token with timestamp, detected language, and score."""

    text: str
    t_start: float
    t_end: float
    language: Optional[str] = None
    score: float = 1.0


@dataclass(frozen=True)
class ErrorInfo:
    """Structured, actionable error for the consumer (FR-015, FR-018, FR-021)."""

    code: str
    message: str
    recoverable: bool = True
    hint: Optional[str] = None


@dataclass(frozen=True)
class PrepareProgress:
    """Progress tick emitted by ``prepare_models`` (FR-011)."""

    asset: str
    downloaded: int
    total: int
    state: ModelState

    @property
    def fraction(self) -> float:
        return self.downloaded / self.total if self.total else 0.0


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #


@dataclass
class Speaker:
    """A distinct voice within one session (session-scoped, anonymous)."""

    label: str
    first_seen: float
    last_seen: float
    total_speech_seconds: float = 0.0


@dataclass
class TranscriptSegment:
    """The core output unit: one attributed unit of recognized speech.

    Ordered by ``start`` then ``end``; finals are non-overlapping per speaker.
    """

    speaker_label: str
    start: float
    end: float
    text: str
    segment_id: str = field(default_factory=lambda: new_id("seg_"))
    language: Optional[str] = None
    confidence: float = 1.0
    confidence_band: Optional[ConfidenceBand] = None
    source: Optional[AudioSourceKind] = None
    is_final: bool = True

    def __post_init__(self) -> None:
        if self.confidence_band is None:
            self.confidence_band = band_for(self.confidence)
        if self.end < self.start:
            # Defensive: never emit an inverted span (consumer relies on end >= start).
            # Warn so an upstream timestamp bug isn't silently hidden.
            import warnings

            warnings.warn(
                f"TranscriptSegment inverted span (start={self.start} > end={self.end}); clamping end=start",
                stacklevel=2,
            )
            self.end = self.start


@dataclass
class AudioSource:
    """A configured/observed capture input."""

    kind: AudioSourceKind
    enabled: bool = True
    state: CaptureState = CaptureState.IDLE
    device_name: Optional[str] = None
    sample_rate_in: Optional[int] = None


@dataclass
class ModelAsset:
    """A downloadable model required by the pipeline (research Decision 6)."""

    name: str
    kind: ModelKind
    framework: ModelFramework
    repo_id: str
    revision: str
    expected_files: Sequence[str] = field(default_factory=tuple)
    cache_path: Optional[str] = None
    state: ModelState = ModelState.ABSENT
    supported_languages: Optional[Sequence[str]] = None

    def is_cached(self) -> bool:
        return self.state in (ModelState.CACHED, ModelState.LOADED)


@dataclass
class SystemReadinessReport:
    """Snapshot of whether the backend can run (FR-013)."""

    models: list = field(default_factory=list)
    mic_permission: bool = False
    system_audio_permission: bool = False
    compute_backend: str = ComputeBackend.CPU.value
    os_supports_system_audio: bool = False
    os_supports_process_tap: bool = False
    missing: list = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.missing and all(m.is_cached() for m in self.models)


__all__ = [
    # enums
    "AudioSourceKind",
    "SessionStatus",
    "CaptureState",
    "ModelKind",
    "ModelFramework",
    "ModelState",
    "ConfidenceBand",
    "ComputeBackend",
    # value types
    "AudioFrame",
    "DiarFrame",
    "AsrToken",
    "ErrorInfo",
    "PrepareProgress",
    # entities
    "Speaker",
    "TranscriptSegment",
    "AudioSource",
    "ModelAsset",
    "SystemReadinessReport",
    # helpers
    "band_for",
    "new_id",
]
