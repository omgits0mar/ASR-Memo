"""Real-model smoke test for the Nemotron ONNX transcriber (needs_models).

Gated on the ASR model being cached (``prepare_models()``); skipped otherwise so
the offline suite stays green without the ~1.2 GB download. Validates that the
3-graph cache-aware RNN-T wiring (encoder + LSTM prediction net + joint), the
log-mel front end, vocab, and language prompt all load and run end-to-end and
emit well-formed :class:`AsrToken`s. WER accuracy is covered by ``make validate``
against committed-out LibriSpeech clips (see tests/fixtures/validation).
"""

from __future__ import annotations

import numpy as np
import pytest

from meeting_asr.asr.nemotron_onnx import NemotronOnnxTranscriber
from meeting_asr.models.registry import (
    check_cached,
    default_cache_dir,
    model_registry,
)
from meeting_asr.types import AudioFrame, AudioSourceKind, ComputeBackend, ModelKind

pytestmark = pytest.mark.needs_models


def _asr_cached() -> bool:
    asset = next(a for a in model_registry() if a.kind is ModelKind.ASR)
    return check_cached(asset, default_cache_dir())


requires_asr = pytest.mark.skipif(not _asr_cached(), reason="ASR model not cached")


@requires_asr
def test_real_transcriber_loads_and_resolves_export():
    tx = NemotronOnnxTranscriber()
    tx.load(ComputeBackend.CPU, precision="fp16")
    # Config + assets resolved from the real export.
    assert tx._blank_id == 13087
    assert len(tx._vocab) >= 13000
    assert tx._resolve_slot("en") is not None
    assert tx._resolve_slot("es") is not None
    # Three sessions opened with the expected I/O.
    assert {i.name for i in tx._enc.get_inputs()} >= {"audio_signal", "language_mask", "pre_cache"}
    assert [o.name for o in tx._joint.get_outputs()] == ["logits"]


@requires_asr
def test_real_transcriber_whole_utterance_emits_wellformed_tokens():
    tx = NemotronOnnxTranscriber()
    tx.load(ComputeBackend.CPU)
    # 2 s of band-limited noise — exercises the full pipeline deterministically.
    rng = np.random.default_rng(0)
    audio = (0.02 * rng.standard_normal(32000)).astype(np.float32)
    tokens = tx.transcribe_array(audio, language="en")
    assert isinstance(tokens, list)
    for tok in tokens:
        assert tok.text
        assert tok.t_end >= tok.t_start >= 0.0
        assert 0.0 <= tok.score <= 1.0
        assert tok.language == "en"


@requires_asr
def test_real_transcriber_streaming_matches_protocol():
    tx = NemotronOnnxTranscriber()
    tx.load(ComputeBackend.CPU)
    rng = np.random.default_rng(1)
    audio = (0.02 * rng.standard_normal(48000)).astype(np.float32)  # 3 s
    step = 1600  # 100 ms frames
    emitted = []
    t = 0.0
    for i in range(0, len(audio), step):
        pcm = audio[i : i + step]
        frame = AudioFrame(pcm=pcm, t_start=t, t_end=t + len(pcm) / 16000,
                           source=AudioSourceKind.MICROPHONE)
        emitted.extend(tx.push(frame, language_hint="en"))
        t += len(pcm) / 16000
    emitted.extend(tx.flush())
    assert isinstance(emitted, list)
    # Timestamps are non-decreasing across the stream.
    starts = [tok.t_start for tok in emitted]
    assert starts == sorted(starts)
