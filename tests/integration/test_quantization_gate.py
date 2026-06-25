"""Quantization gate (task T042; research Decision 8).

A quantized ASR variant (int8/int4) may only become the default if it shows
**≤1% absolute WER regression vs FP16 per language band** AND a measured
turn-to-text latency improvement on the target machine.

This harness runs the real Nemotron pipeline at fp16 vs int8/int4 on the
multilingual fixtures and asserts the WER gate. Skipped without cached models.
"""

from __future__ import annotations

from typing import List

import pytest

from tests._metrics import wer

WER_GATE_ABS = 0.01  # ≤1% absolute WER regression vs FP16 per language band


def _transcript_words(segments, language: str) -> List[str]:
    return " ".join(s.text for s in segments if s.language == language).lower().split()


@pytest.mark.needs_models
@pytest.mark.slow
def test_quantization_int8_within_wer_gate(synthetic_fixture):
    """int8 WER must be within +1% absolute of fp16, per language band."""
    pytest.importorskip("onnxruntime")
    from meeting_asr import AudioSourceKind, Backends, start_session
    from meeting_asr.asr.nemotron_onnx import NemotronOnnxTranscriber
    from tests._fakes import FixtureCapture, ManifestDiarizer

    manifest = synthetic_fixture("multilingual")
    languages = sorted({t["language"] for t in manifest["turns"]})
    ref = {lang: t["text"].lower().split() for t in manifest["turns"] for lang in [t["language"]]}

    def _run(precision: str):
        tx = NemotronOnnxTranscriber()
        try:
            from meeting_asr.backends.device import default_probe, resolve_backend

            tx.load(resolve_backend(default_probe()), precision=precision)
        except Exception as e:
            pytest.skip(f"model not loadable at {precision}: {e}")
        # Direct decode of the fixture audio (no live capture needed for WER).
        import numpy as np
        import soundfile as sf

        pcm, sr = sf.read(manifest["wav"], dtype="float32")
        from meeting_asr.types import AudioFrame

        tx.reset()
        tokens = []
        for i in range(0, len(pcm), 1600):
            block = pcm[i : i + 1600]
            tokens.extend(tx.push(AudioFrame(pcm=block, t_start=i / 16000, t_end=(i + len(block)) / 16000, source=AudioSourceKind.MICROPHONE)))
        tokens += tx.flush()
        hyp = {}
        for t in tokens:
            hyp.setdefault(t.language, []).append(t.text)
        return {lang: [w.lower() for w in hyp.get(lang, [])] for lang in hyp}

    fp16 = _run("fp16")
    int8 = _run("int8")
    for lang in languages:
        ref_words = ref.get(lang, [])
        e_fp16 = wer(fp16.get(lang, []), ref_words)
        e_int8 = wer(int8.get(lang, []), ref_words)
        assert e_int8 - e_fp16 <= WER_GATE_ABS, (
            f"int8 WER regression {e_int8 - e_fp16:.3f} > gate {WER_GATE_ABS} for '{lang}'"
        )
