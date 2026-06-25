"""Offline self-test of the validation harness (task T043 / US5).

Feeds *synthetic* labeled clips through the real pipeline with injected fakes whose
output matches the references (``ManifestDiarizer``/``ManifestTranscriber`` built
from the same turns, latency ≈ 0 so the full timeline emits), then asserts metric
computation + ``ValidationReport`` assembly + reproducibility — no network, no models.
"""

from __future__ import annotations

import pytest

from meeting_asr import Backends, transcribe_file
from meeting_asr.audio.file_capture import FileCapture
from tests._fakes import ManifestDiarizer, ManifestTranscriber
from validation.datasets import ValidationSample
from validation.runner import run_validation


def _samples_from(data):
    wav = data["wav"]
    turns = data["turns"]
    ref_text = " ".join(t["text"] for t in turns)
    return [
        ValidationSample(
            sample_id="multi-asr", path=wav, axis="asr",
            ref_text=ref_text, ref_turns=None, ref_languages=None,
            source="synthetic self-test (002)",
        ),
        ValidationSample(
            sample_id="multi-diar", path=wav, axis="diarization",
            ref_text=None,
            ref_turns=[(t["speaker"], t["t_start"], t["t_end"]) for t in turns],
            ref_languages=None, source="synthetic self-test (002)",
        ),
        ValidationSample(
            sample_id="multi-lang", path=wav, axis="language",
            ref_text=None, ref_turns=None,
            ref_languages=[(t["t_start"], t["t_end"], t["language"]) for t in turns],
            source="synthetic self-test (002)",
        ),
    ]


def _matching_transcribe(turns):
    """transcribe_fn that builds fakes from the same turns → output matches refs."""
    def _fn(sample):
        backends = Backends(
            capture=FileCapture(sample.path),
            diarizer=ManifestDiarizer(turns, latency_s=0.08),
            transcriber=ManifestTranscriber(turns, latency_s=0.0),
        )
        return transcribe_file(sample.path, _backends=backends)
    return _fn


def test_run_validation_assembles_passing_report(synthetic_fixture):
    data = synthetic_fixture("multilingual")
    samples = _samples_from(data)
    report = run_validation(samples, transcribe_fn=_matching_transcribe(data["turns"]))

    assert report.passed is True
    assert len(report.per_clip) == 3
    assert report.aggregate.n_clips == 3

    by_axis = {c.axis: c for c in report.per_clip}
    assert by_axis["asr"].wer is not None and by_axis["asr"].wer <= 0.15
    assert by_axis["asr"].passed is True
    assert by_axis["diarization"].diarization_accuracy is not None
    assert by_axis["diarization"].diarization_accuracy >= 0.90
    assert by_axis["language"].language_id_accuracy is not None
    assert by_axis["language"].language_id_accuracy >= 0.95

    # Thresholds recorded in the report.
    assert report.thresholds["wer_max"] == pytest.approx(0.15)
    assert report.aggregate.mean_wer is not None


def test_run_validation_is_reproducible(synthetic_fixture):
    data = synthetic_fixture("multilingual")
    samples = _samples_from(data)
    fn = _matching_transcribe(data["turns"])
    r1 = run_validation(samples, transcribe_fn=fn)
    r2 = run_validation(samples, transcribe_fn=fn)
    assert r1.passed == r2.passed
    for c1, c2 in zip(r1.per_clip, r2.per_clip):
        for attr in ("wer", "diarization_accuracy", "language_id_accuracy"):
            v1, v2 = getattr(c1, attr), getattr(c2, attr)
            if v1 is None:
                assert v2 is None
            else:
                assert v1 == pytest.approx(v2, abs=1e-6)


def test_run_validation_flags_failures(tmp_path, synthetic_fixture):
    """A mismatched reference → per-clip failure + aggregate not passing."""
    import soundfile as sf
    import numpy as np

    data = synthetic_fixture("single_speaker_en")
    wav = data["wav"]
    # Reference text is deliberately wrong → high WER.
    bad = ValidationSample(
        sample_id="bad-asr", path=wav, axis="asr",
        ref_text="completely unrelated gibberish words here",
        ref_turns=None, ref_languages=None, source="synthetic",
    )
    report = run_validation([bad], transcribe_fn=_matching_transcribe(data["turns"]))
    clip = report.per_clip[0]
    assert clip.wer > 0.5
    assert clip.passed is False
    assert report.passed is False
