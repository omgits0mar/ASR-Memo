"""Fusion aligner (task T022).

Aligns the ``DiarFrame`` timeline with the ``AsrToken`` stream and emits
``TranscriptSegment``s — grouping contiguous tokens that share the dominant
overlapping speaker, carrying token language/score up to the segment.

This is the *fuse-don't-gate* topology (plan.md Complexity Tracking): diarization
and ASR run concurrently; the aligner merges their outputs on the shared session
clock. An alignment buffer holds ASR tokens whose time window the diarizer has not
yet decided (≈ diarization latency) so late speaker decisions are honored.

Pure logic — fully unit-testable offline with synthetic DiarFrames/AsrTokens.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

from .._logging import get_logger
from ..types import AsrToken, DiarFrame, TranscriptSegment, new_id

_log = get_logger("fusion.aligner")


class Aligner:
    """Fuse diarization timeline + ASR token stream into transcript segments.

    Call :meth:`push_diar` as DiarFrames arrive and :meth:`push_tokens` as ASR
    tokens arrive (in any order/timing). Finalized segments are returned; tokens
    whose window is beyond the diarization frontier are buffered until the
    diarizer catches up. :meth:`flush` finalizes everything at end-of-stream.
    """

    # Unattributed-speaker sentinel (diarizer hasn't decided this window yet).
    _UNKNOWN = "Speaker ?"

    def __init__(self, segment_gap_s: float = 1.5, max_segment_s: float = 30.0) -> None:
        # segment_gap_s default 1.5s: natural inter-phrase pauses in connected
        # speech (~0.3-0.8s) should NOT fragment a sentence; only a real pause
        # (turn-taking / topic break) starts a new line.
        self.segment_gap_s = segment_gap_s
        self.max_segment_s = max_segment_s
        self._diar: List[DiarFrame] = []
        self._buffer: List[AsrToken] = []
        self._diar_dirty = True
        self._lock = threading.RLock()  # diar + ASR workers feed concurrently
        # Stable id for the currently-open run so provisional (is_final=False)
        # updates re-render the SAME UI line in place as it grows, then finalize.
        self._open_id: Optional[str] = None
        self._open_speaker: Optional[str] = None

    # ---- diarization side ----

    @property
    def diar_frontier(self) -> float:
        """Latest session time the diarizer has decided (max DiarFrame t_end)."""
        if not self._diar:
            return 0.0
        return max(f.t_end for f in self._diar)

    def push_diar(self, frames: List[DiarFrame]) -> None:
        with self._lock:
            if frames:
                self._diar.extend(frames)
                self._diar_dirty = True

    # ---- token side ----

    def push_tokens(self, tokens: List[AsrToken]) -> List[TranscriptSegment]:
        with self._lock:
            # Drop silence/empty tokens at entry — they never form text and must not
            # close an open run. (silence edge case: emit nothing for non-speech.)
            self._buffer.extend(t for t in tokens if t.text and t.text.strip())
            return self._drain()

    def flush(self) -> List[TranscriptSegment]:
        """Finalize all buffered tokens best-effort (end-of-stream)."""
        with self._lock:
            return self._drain(finalize=True)

    # ---- internals ----

    def _ensure_diar_sorted(self) -> None:
        if self._diar_dirty:
            self._diar.sort(key=lambda f: f.t_start)
            self._diar_dirty = False

    def _dominant_speaker(self, t_start: float, t_end: float) -> Optional[Tuple[str, float]]:
        """Return (speaker_label, weighted_score) with the greatest overlap, or None."""
        self._ensure_diar_sorted()
        if not self._diar:
            return None
        acc: dict[str, float] = {}
        for f in self._diar:
            overlap = min(t_end, f.t_end) - max(t_start, f.t_start)
            if overlap <= 0:
                continue
            acc[f.speaker_label] = acc.get(f.speaker_label, 0.0) + overlap * max(0.0, f.score)
        if not acc:
            return None
        label = max(acc, key=acc.get)
        # representative score: the diar score weighted share, capped to 1.0
        total = sum(acc.values())
        score = min(1.0, acc[label] / total) if total > 0 else 0.0
        return label, score

    def _build_segment(
        self, speaker: str, tokens: List[AsrToken], *, segment_id: str, is_final: bool = True
    ) -> TranscriptSegment:
        text = " ".join(t.text for t in tokens if t.text and t.text.strip())
        langs = [t.language for t in tokens if t.language]
        # majority language; fall back to first
        language = max(set(langs), key=langs.count) if langs else None
        confidence = sum(t.score for t in tokens) / len(tokens) if tokens else 0.0
        # FR-018: unknown/unsupported language (None) → force LOW/UNKNOWN confidence band.
        if language is None:
            confidence = min(confidence, 0.2)
        return TranscriptSegment(
            speaker_label=speaker,
            start=tokens[0].t_start,
            end=tokens[-1].t_end,
            text=text,
            segment_id=segment_id,
            language=language,
            confidence=confidence,
            is_final=is_final,
        )

    def _drain(self, *, finalize: bool = False) -> List[TranscriptSegment]:
        """Fuse buffered tokens into segments.

        Closed runs (speaker change, real gap > ``segment_gap_s``, or
        ``max_segment_s`` reached) emit a **final** segment. The still-open
        trailing run emits a **provisional** (``is_final=False``) segment that
        carries a stable id, so the UI updates the same line live as it grows —
        no waiting for Stop, and natural inter-phrase pauses don't fragment it.
        """
        emitted: List[TranscriptSegment] = []
        if not self._buffer:
            return emitted
        self._buffer.sort(key=lambda t: t.t_start)

        run_speaker: Optional[str] = None
        run_tokens: List[AsrToken] = []
        processed: List[AsrToken] = []
        open_run_held = False  # the trailing run isn't ready to close yet

        for tok in self._buffer:
            ready = finalize or tok.t_end <= self.diar_frontier
            if not ready:
                # Beyond the diar frontier: the diarizer hasn't decided this
                # window. Don't hold the whole line for it — keep it in the open
                # run (provisional) and stop consuming further tokens.
                open_run_held = True
                if run_speaker is None:
                    # Attribute provisionally to the most recent known speaker (or
                    # carry the prior open speaker) so the line still streams.
                    run_speaker = self._open_speaker or self._UNKNOWN
                run_tokens.append(tok)
                break
            dom = self._dominant_speaker(tok.t_start, tok.t_end)
            # Unattributed window → continue the current run rather than break it
            # into a phantom "Speaker ?" line (C). Only stand-alone unknowns at the
            # very start fall back to the sentinel.
            label = dom[0] if dom is not None else (run_speaker or self._UNKNOWN)

            if run_speaker is None or run_speaker == self._UNKNOWN:
                if run_speaker == self._UNKNOWN and dom is not None:
                    run_speaker = label  # adopt the first real attribution
                if run_speaker is None:
                    run_speaker, run_tokens = label, [tok]
                else:
                    run_tokens.append(tok)
            elif (
                label == run_speaker
                and (tok.t_start - run_tokens[-1].t_end) <= self.segment_gap_s
                and (tok.t_end - run_tokens[0].t_start) <= self.max_segment_s
            ):
                run_tokens.append(tok)
            else:
                emitted.append(self._close_run(run_speaker, run_tokens))
                processed.extend(run_tokens)
                run_speaker, run_tokens = label, [tok]

        if run_speaker is not None and run_tokens:
            if finalize and not open_run_held:
                # End of stream: finalize the trailing run.
                emitted.append(self._close_run(run_speaker, run_tokens))
                processed.extend(run_tokens)
            else:
                # Open run continues: emit a provisional (live) update with a
                # stable id so the UI grows the same line in place.
                if self._open_id is None:
                    self._open_id = new_id("seg_")
                self._open_speaker = run_speaker
                emitted.append(
                    self._build_segment(
                        run_speaker, run_tokens, segment_id=self._open_id, is_final=False
                    )
                )

        # Keep unprocessed tokens + the open run's tokens (they re-emit next push).
        processed_set = set(map(id, processed))
        held = [t for t in self._buffer if id(t) not in processed_set]
        self._buffer = held
        return emitted

    def _close_run(self, speaker: str, tokens: List[AsrToken]) -> TranscriptSegment:
        """Finalize an open run: reuse its provisional id (so the UI replaces the
        same line) and clear the open-run state."""
        seg_id = self._open_id or new_id("seg_")
        seg = self._build_segment(speaker, tokens, segment_id=seg_id, is_final=True)
        self._open_id = None
        self._open_speaker = None
        return seg


__all__ = ["Aligner"]
