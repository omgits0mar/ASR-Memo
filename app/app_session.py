"""AppSession — the UI-facing view model around one transcription run (task T009).

Wraps a backend :class:`meeting_asr.TranscriptionSession` (live or file) and projects
its state onto the UI status machine (``data-model.md``)::

    setting_up → ready → (recording | processing) → stopping → stopped
                                                   ↘ any state → error

Holds the accumulated segments, the speaker roster (with deterministic arrival-order
color), file-import progress, and the terminal error. The bridge reads/writes this;
the backend never depends on it (Constitution VII).
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple

from meeting_asr.types import (
    AudioSourceKind,
    ErrorInfo,
    SessionStatus,
    Speaker,
    TranscriptSegment,
    new_id,
)

from .dto import SPEAKER_COLORS, speaker_color

__all__ = ["AppStatus", "InputMode", "AppSession", "map_backend_status"]


class InputMode(str, Enum):
    LIVE = "live"
    FILE = "file"


class AppStatus(str, Enum):
    SETTING_UP = "setting_up"
    READY = "ready"
    STARTING = "starting"
    RECORDING = "recording"
    PROCESSING = "processing"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

    @property
    def active(self) -> bool:
        return self in (AppStatus.STARTING, AppStatus.RECORDING, AppStatus.PROCESSING)


def map_backend_status(status: SessionStatus, input_mode: "InputMode") -> AppStatus:
    """Project a backend ``SessionStatus`` (+ mode) onto a UI ``AppStatus``."""
    if status is SessionStatus.ACTIVE:
        return AppStatus.PROCESSING if input_mode is InputMode.FILE else AppStatus.RECORDING
    if status in (SessionStatus.CREATED, SessionStatus.PREPARING):
        return AppStatus.SETTING_UP
    if status is SessionStatus.STOPPING:
        return AppStatus.STOPPING
    if status is SessionStatus.STOPPED:
        return AppStatus.STOPPED
    return AppStatus.ERROR  # SessionStatus.ERROR


class AppSession:
    """One run's UI state. Thread-safe: fed from pipeline/worker threads, read by the UI."""

    def __init__(
        self,
        *,
        input_mode: InputMode,
        source_kinds: Tuple[AudioSourceKind, ...] = (),
        file_path: Optional[str] = None,
        language_hint: Optional[str] = None,
        app_session_id: Optional[str] = None,
    ) -> None:
        self.app_session_id = app_session_id or new_id("appsess_")
        self.input_mode = input_mode
        self.source_kinds = tuple(source_kinds)
        self.file_path = file_path
        self.language_hint = language_hint
        self.status: AppStatus = AppStatus.SETTING_UP
        self.segments: List[TranscriptSegment] = []
        self._seg_index: Dict[str, int] = {}  # segment_id → position (upsert)
        self.speakers: Dict[str, Speaker] = {}
        self.progress: Optional[float] = None if input_mode is InputMode.LIVE else 0.0
        self.error: Optional[ErrorInfo] = None
        self.backend_session = None  # the wrapped meeting_asr.TranscriptionSession
        self._arrival: Dict[str, int] = {}  # label → arrival index (→ color)
        self._lock = threading.RLock()

    # ---- mutation (called by the bridge) ----

    def begin(self, backend_session) -> None:
        self.backend_session = backend_session
        self._set_status(map_backend_status(SessionStatus.ACTIVE, self.input_mode))

    def claim(self) -> None:
        """Mark the session as actively claimed so the bridge busy guard covers the
        worker-startup window (e.g. file import runs the pipeline on a worker;
        without this, a concurrent start could slip in before PROCESSING emits)."""
        self._set_status(
            AppStatus.PROCESSING if self.input_mode is InputMode.FILE else AppStatus.RECORDING
        )

    def starting(self) -> None:
        """Live session is loading models / starting capture on a worker (STARTING is
        active → busy guard covers the load window; the UI shows a 'starting…' state)."""
        self._set_status(AppStatus.STARTING)

    def ready(self) -> None:
        self._set_status(AppStatus.READY)

    def add_segment(self, seg: TranscriptSegment) -> str:
        """Upsert a segment by id (provisional live updates replace in place), keep
        the roster correct, and return its arrival-order color.

        Live runs stream as provisional (``is_final=False``) updates with a stable
        ``segment_id``; a final later replaces them. Replacing in place keeps one row
        per run in the snapshot/export — matching the UI's segment_id dedupe."""
        with self._lock:
            if seg.speaker_label not in self._arrival:
                self._arrival[seg.speaker_label] = len(self._arrival)
            idx = self._seg_index.get(seg.segment_id)
            if idx is None:
                self._seg_index[seg.segment_id] = len(self.segments)
                self.segments.append(seg)
            else:
                self.segments[idx] = seg
            self._recompute_speakers()
            return speaker_color(self._arrival[seg.speaker_label])

    def _recompute_speakers(self) -> None:
        """Rebuild the roster from the current segments (idempotent under upserts)."""
        roster: Dict[str, Speaker] = {}
        for s in self.segments:
            dur = max(0.0, s.end - s.start)
            spk = roster.get(s.speaker_label)
            if spk is None:
                roster[s.speaker_label] = Speaker(
                    label=s.speaker_label, first_seen=s.start,
                    last_seen=s.end, total_speech_seconds=dur,
                )
            else:
                spk.first_seen = min(spk.first_seen, s.start)
                spk.last_seen = max(spk.last_seen, s.end)
                spk.total_speech_seconds += dur
        self.speakers = roster

    def set_progress(self, fraction: float) -> None:
        with self._lock:
            self.progress = max(self.progress or 0.0, float(fraction))

    def set_error(self, info: ErrorInfo) -> None:
        with self._lock:
            self.error = info
            self._set_status(AppStatus.ERROR)

    def stopping(self) -> None:
        self._set_status(AppStatus.STOPPING)

    def stopped(self) -> None:
        self._set_status(AppStatus.STOPPED)

    # ---- reads (for the bridge/UI/export) ----

    def is_active(self) -> bool:
        with self._lock:
            return self.status.active

    def transcript_snapshot(self) -> List[TranscriptSegment]:
        with self._lock:
            return sorted(self.segments, key=lambda s: (s.start, s.end))

    def speakers_view(self) -> List[Tuple[Speaker, str, int]]:
        """(speaker, color, segment_count) per roster speaker, in arrival order."""
        with self._lock:
            counts: Dict[str, int] = {}
            for seg in self.segments:
                counts[seg.speaker_label] = counts.get(seg.speaker_label, 0) + 1
            ordered = sorted(self.speakers.values(), key=lambda s: self._arrival.get(s.label, 0))
            return [(s, speaker_color(self._arrival[s.label]), counts.get(s.label, 0)) for s in ordered]

    def session_meta(self) -> dict:
        """Lightweight metadata embedded in exports (mode, language hint, speakers)."""
        with self._lock:
            return {
                "app_session_id": self.app_session_id,
                "input_mode": self.input_mode.value,
                "language_hint": self.language_hint,
            }

    # ---- internal ----

    def _set_status(self, status: AppStatus) -> None:
        with self._lock:
            self.status = status
