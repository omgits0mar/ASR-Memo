"""TranscriptionSession — state machine + segment delivery (task T023).

Holds the chronological transcript, the speaker roster, and the session status,
and delivers finalized segments to the consumer via the ``on_segment`` callback
and/or a blocking :meth:`segments` iterator. Thread-safe: the pipeline feeds it
from worker threads while the consumer reads from the main thread.

State transitions (data-model.md)::

    CREATED → PREPARING → ACTIVE → STOPPING → STOPPED
       └──────────────────────────────────────→ ERROR  (from any state)
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Dict, Iterator, List, Optional

from ._logging import get_logger, log_error_info
from .types import (
    AudioSource,
    ErrorInfo,
    SessionStatus,
    Speaker,
    TranscriptSegment,
)

_log = get_logger("session")

_SENTINEL = object()  # wakes the blocking iterator when the stream ends


class TranscriptionSession:
    """One live capture-and-transcribe run."""

    def __init__(
        self,
        *,
        sources: List[AudioSource],
        language_hint: Optional[str] = None,
        on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
        on_error: Optional[Callable[[ErrorInfo], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        from .types import new_id

        self.session_id = session_id or new_id("sess_")
        self.sources = list(sources)
        self.language_hint = language_hint
        self.status: SessionStatus = SessionStatus.CREATED
        self.started_at: Optional[float] = None
        self.stopped_at: Optional[float] = None
        self.error: Optional[ErrorInfo] = None

        self._on_segment = on_segment
        self._on_error = on_error
        self._on_stop = on_stop
        self._segments: List[TranscriptSegment] = []
        self._segment_index: Dict[str, int] = {}  # segment_id → position (upsert)
        self._speakers: Dict[str, Speaker] = {}
        self._q: "queue.Queue[object]" = queue.Queue()
        self._lock = threading.RLock()

    # ---- lifecycle ----

    def begin(self) -> None:
        """Move CREATED → ACTIVE and record the start time."""
        import time as _time

        with self._lock:
            self.started_at = _time.monotonic()
            self.status = SessionStatus.ACTIVE

    def _transition(self, status: SessionStatus) -> None:
        with self._lock:
            self.status = status

    def set_error(self, info: ErrorInfo) -> None:
        """Surface a terminal error (FR-015, FR-018). Moves the session to ERROR."""
        with self._lock:
            self.error = info
            self.status = SessionStatus.ERROR
        self._dispatch_error(info)

    def notify(self, info: ErrorInfo) -> None:
        """Surface a NON-terminal, recoverable condition (FR-021: lag/compute pressure).

        Reports to the consumer via ``on_error`` without changing session status —
        the session stays ACTIVE and keeps the audio timeline coherent.
        """
        log_error_info(_log, info)
        if self._on_error is not None:
            try:
                self._on_error(info)
            except Exception:  # never let a consumer callback kill the pipeline
                _log.exception("on_error callback raised")

    def _dispatch_error(self, info: ErrorInfo) -> None:
        log_error_info(_log, info)
        if self._on_error is not None:
            try:
                self._on_error(info)
            except Exception:
                _log.exception("on_error callback raised")

    # ---- segment delivery (called by the pipeline) ----

    def deliver_segment(self, segment: TranscriptSegment) -> None:
        """Upsert a segment by id (provisional updates replace in place), update
        the speaker roster, and fan out.

        Live segments stream as provisional (``is_final=False``) updates carrying a
        stable ``segment_id`` while a run grows, then a final replaces them — so the
        stored transcript holds one row per run, not one per update.
        """
        with self._lock:
            idx = self._segment_index.get(segment.segment_id)
            if idx is None:
                self._segment_index[segment.segment_id] = len(self._segments)
                self._segments.append(segment)
            else:
                self._segments[idx] = segment  # replace the prior (provisional) version
            self._update_speaker(segment)
        if self._on_segment is not None:
            try:
                self._on_segment(segment)
            except Exception:
                _log.exception("on_segment callback raised")
        self._q.put(segment)

    def _update_speaker(self, segment: TranscriptSegment) -> None:
        # Recompute the roster from the current segment list so provisional→final
        # replacements (same id) don't double-count duration or strand a label from
        # a superseded provisional attribution. Segment counts are small (one row
        # per run), so a full recompute under the lock is cheap and always correct.
        roster: Dict[str, Speaker] = {}
        for seg in self._segments:
            dur = max(0.0, seg.end - seg.start)
            spk = roster.get(seg.speaker_label)
            if spk is None:
                roster[seg.speaker_label] = Speaker(
                    label=seg.speaker_label, first_seen=seg.start,
                    last_seen=seg.end, total_speech_seconds=dur,
                )
            else:
                spk.first_seen = min(spk.first_seen, seg.start)
                spk.last_seen = max(spk.last_seen, seg.end)
                spk.total_speech_seconds += dur
        self._speakers = roster

    # ---- consumer API ----

    def segments(self) -> Iterator[TranscriptSegment]:
        """Blocking iterator yielding finalized segments in arrival order.

        Ends (StopIteration) once the session is terminal and the queue is drained.
        """
        while True:
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                if self.status in (SessionStatus.STOPPED, SessionStatus.ERROR):
                    continue  # drain any tail, then the sentinel closes it
                continue
            if item is _SENTINEL:
                return
            assert isinstance(item, TranscriptSegment)
            yield item

    def transcript(self) -> List[TranscriptSegment]:
        """Snapshot of all finalized segments so far, chronological (FR-019)."""
        with self._lock:
            return sorted(self._segments, key=lambda s: (s.start, s.end))

    def speakers(self) -> Dict[str, Speaker]:
        """Current speaker roster (stable labels)."""
        with self._lock:
            return dict(self._speakers)

    def stop(self, *, timeout_s: float = 10.0) -> List[TranscriptSegment]:
        """Halt capture/processing, flush, return the final transcript. Idempotent."""
        import time as _time

        with self._lock:
            if self.status in (SessionStatus.STOPPED, SessionStatus.ERROR):
                self._q.put(_SENTINEL)
                return self.transcript()
            self.status = SessionStatus.STOPPING

        if self._on_stop is not None:
            try:
                self._on_stop()
            except Exception:
                _log.exception("on_stop raised; forcing stop")

        with self._lock:
            self.stopped_at = _time.monotonic()
            self.status = SessionStatus.STOPPED
        self._q.put(_SENTINEL)
        return self.transcript()

    def signal_end(self) -> None:
        """Tell the iterator the live stream has ended (no error)."""
        self._q.put(_SENTINEL)


__all__ = ["TranscriptionSession"]
