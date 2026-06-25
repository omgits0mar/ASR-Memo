"""Validation metrics — WER, diarization accuracy, language-ID accuracy (T044 / US5).

Pure functions over plain Python / :class:`~meeting_asr.TranscriptSegment`. WER uses
``jiwer`` when available (dev extra) with a deterministic pure-Python Levenshtein
fallback so the harness runs offline without it. Diarization accuracy is
permutation-invariant (optimal reference↔hypothesis label matching). See
``contracts/validation_report.md``.
"""

from __future__ import annotations

from itertools import permutations
from typing import List, Optional, Sequence, Tuple

from meeting_asr.types import TranscriptSegment

__all__ = ["wer", "diarization_accuracy", "der", "language_id_accuracy"]


# --------------------------------------------------------------------------- #
# WER
# --------------------------------------------------------------------------- #


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _levenshtein(ref: List[str], hyp: List[str]) -> int:
    """Word-level edit distance between ref and hyp."""
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, hw in enumerate(hyp, start=1):
            cost = 0 if rw == hw else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate over normalized text (jiwer if present, else Levenshtein)."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    try:  # dev extra; deterministic fallback otherwise
        import jiwer

        return float(jiwer.wer(ref, hyp))
    except ImportError:
        ref_words, hyp_words = ref.split(), hyp.split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        return _levenshtein(ref_words, hyp_words) / len(ref_words)


# --------------------------------------------------------------------------- #
# Diarization accuracy (permutation-invariant)
# --------------------------------------------------------------------------- #

Turn = Tuple[str, float, float]  # (speaker_label, t_start, t_end)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def diarization_accuracy(ref_turns: Sequence[Turn], hyp_segments: Sequence[TranscriptSegment]) -> float:
    """Fraction of reference speech-time correctly attributed, under the optimal
    reference↔hypothesis label permutation (labels are arbitrary ``Speaker N``).

    ``ref_turns`` are ``(speaker, t_start, t_end)``. Returns 0..1; 1.0 is a perfect
    attribution up to label renaming.
    """
    if not ref_turns:
        return 0.0
    ref_speakers = sorted({t[0] for t in ref_turns})
    hyp_labels = sorted({s.speaker_label for s in hyp_segments})
    total_ref = sum(max(0.0, t[2] - t[1]) for t in ref_turns)
    if total_ref <= 0.0 or not hyp_labels:
        return 0.0

    # overlap[ref_speaker][hyp_label] = total co-occurring speech time.
    overlap = {r: {h: 0.0 for h in hyp_labels} for r in ref_speakers}
    for rspk, rt0, rt1 in ref_turns:
        for seg in hyp_segments:
            if seg.speaker_label not in overlap[rspk]:
                continue
            ov = _overlap(rt0, rt1, seg.start, seg.end)
            if ov > 0.0:
                overlap[rspk][seg.speaker_label] += ov

    # Optimal injective assignment: each hyp label → a distinct ref speaker.
    best = 0.0
    k = len(hyp_labels)
    for assignment in permutations(ref_speakers, k):  # ref speakers chosen for each hyp label
        total = sum(overlap[assignment[i]][hyp_labels[i]] for i in range(k))
        if total > best:
            best = total
    return best / total_ref


def der(ref_turns: Sequence[Turn], hyp_segments: Sequence[TranscriptSegment]) -> float:
    """Diarization error rate — the error complement of :func:`diarization_accuracy`,
    reported alongside it (``contracts/validation_report.md``). 0.0 = perfect
    attribution (under optimal label matching); higher is worse.
    """
    return 1.0 - diarization_accuracy(ref_turns, hyp_segments)


# --------------------------------------------------------------------------- #
# Language-ID accuracy
# ---------------------------------------------------------------------------


def language_id_accuracy(
    ref_languages: Sequence[Tuple[float, float, str]],
    hyp_segments: Sequence[TranscriptSegment],
) -> float:
    """Fraction of language-bearing hyp segments whose detected language matches the
    reference language for the maximally-overlapping reference span.

    ``ref_languages`` are ``(t_start, t_end, lang)``. Segments with ``language=None``
    are ignored (denominator excludes them).
    """
    scored = [s for s in hyp_segments if s.language]
    if not scored or not ref_languages:
        return 0.0
    correct = 0
    for seg in scored:
        best_lang: Optional[str] = None
        best_ov = 0.0
        for rt0, rt1, rlang in ref_languages:
            ov = _overlap(seg.start, seg.end, rt0, rt1)
            if ov > best_ov:
                best_ov = ov
                best_lang = rlang
        if best_lang is not None and best_lang == seg.language:
            correct += 1
    return correct / len(scored)
