"""Unit tests for validation metrics — WER / diarization / language-ID (T042 / US5).

Known-input correctness checks: WER (via jiwer with a pure-Python fallback),
permutation-invariant diarization accuracy, and per-segment language-ID accuracy.
Offline; jiwer need not be installed (the fallback is exercised equivalently).
"""

from __future__ import annotations

import pytest

from meeting_asr.types import TranscriptSegment
from validation.metrics import der, diarization_accuracy, language_id_accuracy, wer


def _seg(label, start, end, language=None):
    return TranscriptSegment(speaker_label=label, start=start, end=end, text="x", language=language)


# ---- WER ----

def test_wer_identical_is_zero():
    assert wer("hello world", "hello world") == pytest.approx(0.0)


def test_wer_one_deletion():
    assert wer("hello world", "hello") == pytest.approx(0.5)


def test_wer_one_substitution():
    assert wer("the cat sat", "the bat sat") == pytest.approx(1 / 3)


def test_wer_completely_wrong_is_one():
    assert wer("a b c", "x y z") == pytest.approx(1.0)


def test_wer_empty_reference():
    assert wer("", "") == pytest.approx(0.0)
    assert wer("", "extra") == pytest.approx(1.0)


# ---- Diarization accuracy (permutation-invariant) ----

def test_diarization_perfect():
    ref = [("Alice", 0.0, 1.0), ("Bob", 1.0, 2.0)]
    hyp = [_seg("Speaker 1", 0.0, 1.0), _seg("Speaker 2", 1.0, 2.0)]
    assert diarization_accuracy(ref, hyp) == pytest.approx(1.0)


def test_diarization_permutation_invariant():
    # Hyp labels are arbitrary and swapped vs reference → still perfect.
    ref = [("Alice", 0.0, 1.0), ("Bob", 1.0, 2.0)]
    hyp = [_seg("Speaker 2", 0.0, 1.0), _seg("Speaker 1", 1.0, 2.0)]
    assert diarization_accuracy(ref, hyp) == pytest.approx(1.0)


def test_diarization_partial_attribution():
    ref = [("Alice", 0.0, 2.0)]
    hyp = [_seg("Speaker 1", 0.0, 1.0)]  # only half the speech attributed
    assert diarization_accuracy(ref, hyp) == pytest.approx(0.5)


def test_diarization_wrong_speaker_below_one():
    ref = [("Alice", 0.0, 1.0), ("Bob", 1.0, 2.0)]
    # Both ref spans attributed to a single hyp speaker → one ref unmatched.
    hyp = [_seg("Speaker 1", 0.0, 2.0)]
    assert diarization_accuracy(ref, hyp) < 1.0


def test_der_is_complement_of_diarization_accuracy():
    ref = [("Alice", 0.0, 1.0), ("Bob", 1.0, 2.0)]
    hyp = [_seg("Speaker 1", 0.0, 1.0), _seg("Speaker 2", 1.0, 2.0)]
    assert der(ref, hyp) == pytest.approx(0.0)  # perfect → 0 error
    bad = [_seg("Speaker 1", 0.0, 2.0)]
    assert der(ref, bad) == pytest.approx(1.0 - diarization_accuracy(ref, bad))


# ---- Language-ID accuracy ----

def test_language_id_perfect():
    ref = [(0.0, 1.0, "en"), (1.0, 2.0, "es")]
    hyp = [_seg("S", 0.1, 0.9, "en"), _seg("S", 1.1, 1.9, "es")]
    assert language_id_accuracy(ref, hyp) == pytest.approx(1.0)


def test_language_id_partial():
    ref = [(0.0, 1.0, "en"), (1.0, 2.0, "es")]
    hyp = [_seg("S", 0.1, 0.9, "en"), _seg("S", 1.1, 1.9, "fr")]
    assert language_id_accuracy(ref, hyp) == pytest.approx(0.5)


def test_language_id_ignores_no_language_segments():
    ref = [(0.0, 2.0, "en")]
    hyp = [_seg("S", 0.1, 0.9, "en"), _seg("S", 1.1, 1.9, None)]
    # Only the language-bearing segment is scored → 1/1.
    assert language_id_accuracy(ref, hyp) == pytest.approx(1.0)
