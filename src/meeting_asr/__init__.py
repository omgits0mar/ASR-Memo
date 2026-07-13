"""meeting_asr — on-device, realtime, diarized, multilingual meeting transcription.

In-process Python library (no UI / no server). Public surface:

    prepare_models(progress=, force=) -> SystemReadinessReport
    check_readiness()                 -> SystemReadinessReport
    start_session(sources=, language_hint=, on_segment=, on_error=) -> TranscriptionSession

See ``specs/001-meeting-asr-backend/contracts/public_api.md`` for the contract.
"""

from __future__ import annotations

import platform
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ._logging import (
    CapturePermissionError,
    MeetingAsrError,
    ModelError,
    ReadinessError,
    SessionBusyError,
    configure_logging,
    get_logger,
)
from .asr.transcriber import SpeechTranscriber
from .audio.capture import AudioCapture
from .audio.coreaudio_tap import CoreAudioTapCapture
from .audio.microphone import MicrophoneCapture
from .audio.mixer import CompositeCapture
from .backends.device import default_probe, resolve_backend
from .backends.factory import build_inference_backends
from .diarization.diarizer import SpeakerDiarizer
from .fusion.aligner import Aligner
from .models import readiness as _readiness
from .models.readiness import build_readiness
from .models.registry import model_registry
from .models.registry import prepare as _prepare_assets
from .pipeline import Pipeline
from .session import TranscriptionSession
from .types import (
    AudioSource,
    AudioSourceKind,
    CaptureState,
    ComputeBackend,
    ErrorInfo,
    ModelAsset,
    ModelFramework,
    ModelKind,
    ModelState,
    PrepareProgress,
    SessionStatus,
    Speaker,
    SystemReadinessReport,
    TranscriptSegment,
)

_log = get_logger("facade")
configure_logging()

# Public type re-exports.
__all__ = [
    "prepare_models",
    "check_readiness",
    "start_session",
    "transcribe_file",
    "TranscriptionSession",
    "Backends",
    # types
    "AudioSource",
    "AudioSourceKind",
    "CaptureState",
    "ComputeBackend",
    "ErrorInfo",
    "MeetingAsrError",
    "ModelError",
    "ReadinessError",
    "SessionBusyError",
    "CapturePermissionError",
    "ModelAsset",
    "ModelFramework",
    "ModelKind",
    "ModelState",
    "PrepareProgress",
    "SessionStatus",
    "Speaker",
    "SystemReadinessReport",
    "TranscriptSegment",
]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def prepare_models(
    *,
    progress: Optional[Callable[[PrepareProgress], None]] = None,
    force: bool = False,
) -> SystemReadinessReport:
    """Ensure ASR + diarizer models are downloaded and cached (FR-011, FR-012, SC-005).

    Idempotent; loads from cache without re-downloading when present. Resumable on
    interruption; never leaves a corrupt cache. Touches the network ONLY here.
    """
    backend = resolve_backend(default_probe())
    _prepare_assets(model_registry(backend), progress=progress, force=force)
    return check_readiness()


def check_readiness() -> SystemReadinessReport:
    """Report models + permissions + compute + a `missing` list (FR-013). Never raises."""
    return build_readiness()


@dataclass
class Backends:
    """Injectable backend set for ``start_session`` (testing / advanced wiring)."""

    capture: AudioCapture
    diarizer: SpeakerDiarizer
    transcriber: SpeechTranscriber


# Sequential-session guard (FR-020): at most one ACTIVE session over the lifetime.
_active_lock = threading.Lock()
_active_session: Optional[TranscriptionSession] = None


def start_session(
    *,
    sources: Sequence[AudioSourceKind] = (AudioSourceKind.MICROPHONE,),
    language_hint: Optional[str] = None,
    on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
    on_error: Optional[Callable[[ErrorInfo], None]] = None,
    _backends: Optional[Backends] = None,
) -> TranscriptionSession:
    """Create + start a live session (FR-006, FR-010, FR-019, FR-020).

    Begins capture, diarization, and transcription. ``on_segment`` fires per
    finalized segment. Raises ``SessionBusyError`` if a session is already ACTIVE,
    ``ReadinessError`` if required models are missing, or ``CapturePermissionError``
    if a source's permission is denied (with an actionable hint).

    ``_backends`` is an internal/testing seam to inject capture/diarizer/transcriber
    (e.g. fakes); omit it for the production mic + Nemotron + Sortformer path.
    """
    global _active_session

    with _active_lock:
        if _active_session is not None and _active_session.status == SessionStatus.ACTIVE:
            raise SessionBusyError(
                "a session is already ACTIVE; call session.stop() before starting another (FR-020)"
            )

        source_objs = [AudioSource(kind=k, enabled=True) for k in sources]
        backends = _backends or _build_default_backends(sources)

        session = TranscriptionSession(
            sources=source_objs,
            language_hint=language_hint,
            on_segment=on_segment,
            on_error=on_error,
        )
        aligner = Aligner()
        pipeline = Pipeline(
            capture=backends.capture,
            diarizer=backends.diarizer,
            transcriber=backends.transcriber,
            aligner=aligner,
            session=session,
            language_hint=language_hint,
        )
        # Wire session.stop() → pipeline.stop(); clear the active-session slot on stop.
        def _on_stop() -> None:
            try:
                pipeline.stop()
            finally:
                _clear_active()

        session._on_stop = _on_stop  # type: ignore[attr-defined]

        try:
            backend = resolve_backend(default_probe())
            backends.diarizer.load(backend)
            backends.transcriber.load(backend, precision="fp16")
            pipeline.start()
        except CapturePermissionError:
            session._on_stop = None  # type: ignore[attr-defined]
            raise
        except ModelError as e:
            session._on_stop = None  # type: ignore[attr-defined]
            raise ReadinessError(str(e)) from e

        _active_session = session
        return session


def _clear_active() -> None:
    global _active_session
    with _active_lock:
        _active_session = None


def transcribe_file(
    path: str,
    *,
    language_hint: Optional[str] = None,
    on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
    on_error: Optional[Callable[[ErrorInfo], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    _backends: Optional[Backends] = None,
) -> TranscriptionSession:
    """Transcribe an audio file to completion (US2; research Decision 3).

    Same diarize ∥ transcribe → fuse pipeline as :func:`start_session` — the result
    is identical whether audio arrives live or from a file. Builds a
    :class:`~meeting_asr.audio.file_capture.FileCapture` + the default
    diarizer/transcriber (or injected ``_backends``), runs the file to completion,
    then stops — returning the session in ``STOPPED`` (transcript fully available)
    or ``ERROR`` with an :class:`ErrorInfo` for unreadable input.

    Honors the same readiness/busy rules as :func:`start_session`
    (``ReadinessError`` when models are missing, ``SessionBusyError`` when a session
    is already active). ``on_progress`` reports a monotonic 0..1 fraction derived
    from samples consumed; reaches ``1.0`` at completion. ``language_hint=None``
    ⇒ per-turn auto-detection (Principle VI).
    """
    import time as _time

    from .audio.file_capture import FileCapture, FileCaptureError

    if _backends is None:
        try:
            capture: AudioCapture = FileCapture(path)
        except FileCaptureError as e:
            return _error_session(e.info, on_error=on_error)
        backend = resolve_backend(default_probe())
        diarizer, transcriber = build_inference_backends(backend)
        backends = Backends(
            capture=capture,
            diarizer=diarizer,
            transcriber=transcriber,
        )
    else:
        capture = _backends.capture
        backends = _backends

    session = start_session(
        sources=(AudioSourceKind.MICROPHONE,),
        language_hint=language_hint,
        on_segment=on_segment,
        on_error=on_error,
        _backends=backends,
    )

    # Drive the file to completion (poll the capture's progress seam), then stop to
    # flush the aligner + join workers → the transcript is fully finalized.
    frac_fn = getattr(capture, "consumed_fraction", None)
    deadline = _time.monotonic() + 600.0
    last = -1.0
    while _time.monotonic() < deadline:
        if frac_fn is not None:
            frac = min(1.0, max(0.0, float(frac_fn())))
            if on_progress is not None and frac != last:
                on_progress(frac)
                last = frac
            if frac >= 1.0 and capture.state() != CaptureState.CAPTURING:
                break
        elif capture.state() != CaptureState.CAPTURING:
            break
        _time.sleep(0.02)
    if on_progress is not None:
        on_progress(1.0)

    session.stop()
    return session


def _error_session(info: ErrorInfo, *, on_error: Optional[Callable[[ErrorInfo], None]]) -> TranscriptionSession:
    """Build a terminal ERROR session carrying ``info`` (e.g. unreadable file)."""
    session = TranscriptionSession(
        sources=[AudioSource(kind=AudioSourceKind.MICROPHONE, enabled=True)],
        on_error=on_error,
    )
    session.set_error(info)  # → ERROR status + dispatch on_error
    return session


def _build_default_backends(sources: Sequence[AudioSourceKind]) -> Backends:
    """Build production backends. MIC always; SYSTEM added via CompositeCapture (T032)."""
    kinds = set(sources)
    if AudioSourceKind.SYSTEM in kinds:
        if not _readiness.os_supports_system_audio():
            raise ReadinessError(
                "system-audio capture is unavailable on this host; use "
                "sources=(AudioSourceKind.MICROPHONE,) or configure OS loopback"
            )
        capture: AudioCapture = CompositeCapture(
            [MicrophoneCapture(), _system_audio_capture()]
        )
    else:
        capture = MicrophoneCapture()
    backend = resolve_backend(default_probe())
    diarizer, transcriber = build_inference_backends(backend)
    return Backends(
        capture=capture,
        diarizer=diarizer,
        transcriber=transcriber,
    )


def _system_audio_capture() -> AudioCapture:
    system = platform.system()
    if system == "Darwin":
        return CoreAudioTapCapture()
    if system == "Windows":
        from .audio.wasapi_loopback import WasapiLoopbackCapture

        return WasapiLoopbackCapture()
    if system == "Linux":
        from .audio.pipewire_loopback import PipeWireLoopbackCapture

        return PipeWireLoopbackCapture()
    raise ReadinessError(f"system-audio capture is not supported on {system or 'this OS'}")
