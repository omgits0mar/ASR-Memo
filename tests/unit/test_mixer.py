"""Unit tests for audio/mixer.py (T044 scope; T012 single-source path).

Run fully offline (numpy + soxr only) — no capture device, no models.
"""

from __future__ import annotations

import numpy as np
import pytest

from meeting_asr.audio.mixer import (
    SAMPLE_RATE,
    AudioMixer,
    normalize_block,
    resample,
    to_mono,
)
from meeting_asr.types import AudioSourceKind


def _sin(n, rate, freq=440.0):
    t = np.arange(n, dtype=np.float32) / rate
    return np.sin(2 * np.pi * freq * t, dtype=np.float32)


class TestResamplePrimitives:
    def test_resample_48k_to_16k_thirds_length(self):
        out = resample(_sin(4800, 48000), 48000, 16000)
        assert out.dtype == np.float32
        assert abs(len(out) - 1600) <= 2

    def test_resample_8k_to_16k_doubles_length(self):
        out = resample(_sin(800, 8000), 8000, 16000)
        assert abs(len(out) - 1600) <= 2

    def test_to_mono_collapses_channels_by_averaging(self):
        stereo = np.stack([np.full(100, 0.5, np.float32), np.full(100, -0.5, np.float32)], axis=1)
        mono = to_mono(stereo)
        assert mono.shape == (100,) and mono.dtype == np.float32
        assert np.allclose(mono, 0.0, atol=1e-6)

    def test_to_mono_passes_through_mono(self):
        assert to_mono(_sin(100, 16000)).shape == (100,)

    def test_normalize_block_scales_int16_and_downmixes(self):
        loud = np.full((100, 2), 32767, dtype=np.int16)
        out = normalize_block(loud, sample_rate=16000, channels=2)
        assert out.dtype == np.float32 and out.shape == (100,)
        assert np.allclose(out, 1.0, atol=1e-3)


class TestAudioMixerSingleSource:
    def test_feed_produces_continuous_monotonic_clock(self):
        mixer = AudioMixer()
        block = _sin(1600, 16000)  # 0.1 s @ 16 kHz
        f1 = mixer.feed(block, 16000, source=AudioSourceKind.MICROPHONE)
        f2 = mixer.feed(block, 16000, source=AudioSourceKind.MICROPHONE)
        f3 = mixer.feed(block, 16000, source=AudioSourceKind.MICROPHONE)

        assert f1.t_start == 0.0
        assert f1.t_end == pytest.approx(0.1)
        assert f2.t_start == pytest.approx(0.1) and f2.t_start == f1.t_end
        assert f3.t_start == f2.t_end == pytest.approx(0.2)
        assert f3.t_end == pytest.approx(0.3)

    def test_feed_preserves_source_tag_and_dtype(self):
        mixer = AudioMixer()
        f = mixer.feed(_sin(800, 16000), 16000, source=AudioSourceKind.SYSTEM)
        assert f.source is AudioSourceKind.SYSTEM
        assert f.pcm.dtype == np.float32
        assert f.n_samples == 800

    def test_feed_resamples_high_rate_input_to_16k(self):
        mixer = AudioMixer()
        f = mixer.feed(_sin(4800, 48000), 48000, source=AudioSourceKind.MICROPHONE)
        assert f.duration == pytest.approx(0.1)
        assert abs(f.n_samples - 1600) <= 2

    def test_target_rate_constant(self):
        assert SAMPLE_RATE == 16000
