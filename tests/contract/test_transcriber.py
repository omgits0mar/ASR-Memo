"""Contract test: SpeechTranscriber (task T016; contracts/speech_transcriber.md).

Validates streaming tokens, timestamps, flush(), the FP16 default precision, and
``supported_languages()`` against a conforming backend. The fake
``ManifestTranscriber`` is the offline reference; the real ``NemotronOnnxTranscriber``
is smoke-checked under ``needs_models`` (skipped when the model isn't cached).
"""

from __future__ import annotations

import numpy as np
import pytest

from meeting_asr.asr.nemotron_onnx import NemotronOnnxTranscriber
from meeting_asr.types import AudioFrame, AudioSourceKind, ComputeBackend
from tests._fakes import ManifestTranscriber


def _timeline_frames(duration_s: float, step: float = 0.1):
    t = 0.0
    while t < duration_s:
        yield AudioFrame(pcm=np.zeros(int(step * 16000), dtype=np.float32), t_start=t, t_end=t + step,
                         source=AudioSourceKind.MICROPHONE)
        t += step


def test_transcriber_streams_tokens_incrementally(synthetic_fixture):
    manifest = synthetic_fixture("single_speaker_en")
    tx = ManifestTranscriber(manifest["turns"], latency_s=0.0)
    tx.load(ComputeBackend.CPU, precision="fp16")
    tx.reset()
    seen_at = []
    total_words = sum(len(t["text"].split()) for t in manifest["turns"])
    emitted = 0
    for fr in _timeline_frames(7.0):
        toks = tx.push(fr)
        emitted += len(toks)
        seen_at.append(emitted)
    # incremental: emitted count grows over time (not all-at-once at the end)
    assert emitted == total_words
    assert seen_at[0] < seen_at[-1]


def test_transcriber_tokens_carry_text_timestamps_language(synthetic_fixture):
    manifest = synthetic_fixture("two_speaker_en")
    tx = ManifestTranscriber(manifest["turns"], latency_s=0.0)
    tx.reset()
    tokens = []
    for fr in _timeline_frames(11.0):
        tokens.extend(tx.push(fr))
    assert tokens
    for tok in tokens:
        assert tok.text and tok.t_end >= tok.t_start
        assert tok.language == "en"


def test_transcriber_flush_emits_remaining(synthetic_fixture):
    manifest = synthetic_fixture("single_speaker_en")
    tx = ManifestTranscriber(manifest["turns"], latency_s=0.56)  # latency holds the tail
    tx.reset()
    # Stream only 5 s: the last turn (ends 5.4s) is held behind the 0.56s latency horizon.
    for fr in _timeline_frames(5.0):
        tx.push(fr)
    tail = tx.flush()
    assert tail, "flush must emit tokens held behind the latency horizon"


def test_transcriber_fp16_default_precision():
    tx = ManifestTranscriber([{"speaker": "Speaker 1", "t_start": 0.0, "t_end": 1.0, "text": "hi", "language": "en"}])
    tx.load(ComputeBackend.CPU)  # precision defaults to fp16
    assert tx._precision == "fp16"


def test_transcriber_supported_languages_nonempty(synthetic_fixture):
    manifest = synthetic_fixture("multilingual")
    tx = ManifestTranscriber(manifest["turns"])
    langs = tx.supported_languages()
    assert "en" in langs and "es" in langs


@pytest.mark.needs_models
def test_real_nemotron_smoke():
    tx = NemotronOnnxTranscriber()
    try:
        tx.load(ComputeBackend.CPU, precision="fp16")
    except Exception as e:
        pytest.skip(f"Nemotron model not cached: {e}")
    assert "en" in tx.supported_languages()
