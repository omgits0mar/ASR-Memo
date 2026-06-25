"""Pipeline orchestration (task T024).

Topology (research Decision 4, plan.md Complexity Tracking)::

    capture → mix ──┬── diarize (Sortformer)  ──┐
                    └── transcribe (Nemotron) ──┴── fusion.aligner → session

Diarization and ASR run **concurrently** on the same 16 kHz mono stream (each
keeps its own streaming cache) and are fused by timestamp. This preserves
cache-aware streaming for the RNNT decoder (feeding pre-cut per-speaker segments
would reset its cache — rejected for latency/accuracy).

Backpressure (FR-016, FR-021): frames are buffered (never dropped) on unbounded
per-stage queues; if a stage falls behind realtime beyond ``max_lag_s``, a single
``COMPUTE_PRESSURE`` :class:`ErrorInfo` is surfaced to the consumer and the
timeline keeps its coherence.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

from ._logging import get_logger, lag_error_info
from .audio.capture import AudioCapture
from .diarization.diarizer import SpeakerDiarizer
from .asr.transcriber import SpeechTranscriber
from .fusion.aligner import Aligner
from .session import TranscriptionSession
from .types import AudioFrame

_log = get_logger("pipeline")

_STOP = object()


class Pipeline:
    """Drives capture → (diarize ∥ transcribe) → fuse → session delivery."""

    def __init__(
        self,
        *,
        capture: AudioCapture,
        diarizer: SpeakerDiarizer,
        transcriber: SpeechTranscriber,
        aligner: Aligner,
        session: TranscriptionSession,
        language_hint: Optional[str] = None,
        max_lag_s: float = 5.0,
    ) -> None:
        self._capture = capture
        self._diarizer = diarizer
        self._transcriber = transcriber
        self._aligner = aligner
        self._session = session
        self._language_hint = language_hint
        self._max_lag_s = max_lag_s

        self._diar_q: "queue.Queue[object]" = queue.Queue()
        self._asr_q: "queue.Queue[object]" = queue.Queue()
        self._stop = threading.Event()
        self._threads: list = []
        self._lag_signaled = False
        self._frames_buffered = 0

    # ---- lifecycle ----

    def start(self) -> None:
        self._session.begin()
        self._diarizer.reset()
        self._transcriber.reset()
        self._capture.start(self._on_frame)
        self._threads = [
            threading.Thread(target=self._diar_worker, name="diar", daemon=True),
            threading.Thread(target=self._asr_worker, name="asr", daemon=True),
        ]
        for t in self._threads:
            t.start()
        _log.info("pipeline started (concurrent diarize ∥ transcribe → fuse)")

    def stop(self, *, timeout_s: float = 10.0) -> None:
        """Flush in-flight audio, join workers, stop capture (idempotent)."""
        if self._stop.is_set():
            return
        self._stop.set()
        self._diar_q.put(_STOP)
        self._asr_q.put(_STOP)
        for t in self._threads:
            t.join(timeout=timeout_s)

        # Flush the ASR tail through the aligner.
        try:
            tail = self._transcriber.flush()
            for seg in self._aligner.push_tokens(tail):
                self._session.deliver_segment(seg)
            for seg in self._aligner.flush():
                self._session.deliver_segment(seg)
        except Exception:
            _log.exception("error flushing aligner on stop")

        try:
            self._capture.stop()
        except Exception:
            _log.exception("error stopping capture")
        _log.info("pipeline stopped")

    # ---- capture fan-out (called from the capture thread) ----

    def _on_frame(self, frame: AudioFrame) -> None:
        if self._stop.is_set():
            return
        # Buffer on unbounded queues: never drop a frame (FR-016/FR-021).
        self._diar_q.put(frame)
        self._asr_q.put(frame)
        self._frames_buffered += 1
        # Lag signaling: if either queue depth implies we're >max_lag behind, surface once.
        depth = max(self._diar_q.qsize(), self._asr_q.qsize())
        approx_lag = depth * 0.1  # ~100 ms frames; coarse
        if approx_lag > self._max_lag_s and not self._lag_signaled:
            self._lag_signaled = True
            # Non-terminal: signal the constraint, keep the session ACTIVE (FR-021).
            self._session.notify(
                lag_error_info(f"~{approx_lag:.1f}s behind realtime (buffered, not dropped)")
            )

    # ---- workers ----

    def _drain_until_stop(self, q: "queue.Queue[object]"):
        """Yield frames until the stop sentinel is seen."""
        while True:
            item = q.get()
            if item is _STOP:
                return
            assert isinstance(item, AudioFrame)
            yield item

    def _diar_worker(self) -> None:
        try:
            for frame in self._drain_until_stop(self._diar_q):
                try:
                    decisions = self._diarizer.push(frame)
                    if decisions:
                        self._aligner.push_diar(decisions)
                except Exception:
                    _log.exception("diarizer error; continuing")
        except Exception:
            _log.exception("diar worker crashed")

    def _asr_worker(self) -> None:
        try:
            for frame in self._drain_until_stop(self._asr_q):
                try:
                    tokens = self._transcriber.push(frame, language_hint=self._language_hint)
                    if tokens:
                        for seg in self._aligner.push_tokens(tokens):
                            self._session.deliver_segment(seg)
                except Exception:
                    _log.exception("transcriber error; continuing")
        except Exception:
            _log.exception("asr worker crashed")


__all__ = ["Pipeline"]
