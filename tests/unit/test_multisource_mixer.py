"""Unit tests for MultiSourceMixer / CompositeCapture (T031; FR-009)."""

from __future__ import annotations

import numpy as np
import pytest

from meeting_asr.audio.mixer import MultiSourceMixer
from meeting_asr.types import AudioFrame, AudioSourceKind


def _frame(pcm, t0, t1, src):
    return AudioFrame(pcm=np.asarray(pcm, dtype=np.float32), t_start=t0, t_end=t1, source=src)


class TestMultiSourceMixer:
    def test_single_source_passes_through_tagged(self):
        m = MultiSourceMixer(chunk_seconds=0.1)
        pcm = np.full(1600, 0.5, dtype=np.float32)  # 0.1 s
        m.feed(_frame(pcm, 0.0, 0.1, AudioSourceKind.SYSTEM))
        m.feed(_frame(pcm, 0.1, 0.2, AudioSourceKind.SYSTEM))  # advances max_idx → emit chunk 0
        out = m.flush()
        assert out, "expected merged chunks"
        assert all(f.source is AudioSourceKind.SYSTEM for f in out)
        # contiguous timeline
        for a, b in zip(out, out[1:]):
            assert b.t_start == pytest.approx(a.t_end)

    def test_overlapping_sources_sum_and_tag_dominant(self):
        m = MultiSourceMixer(chunk_seconds=0.1)
        loud_mic = np.full(1600, 0.8, dtype=np.float32)   # mic louder
        quiet_sys = np.full(1600, 0.1, dtype=np.float32)
        out: list = []
        out += m.feed(_frame(loud_mic, 0.0, 0.1, AudioSourceKind.MICROPHONE))
        out += m.feed(_frame(quiet_sys, 0.0, 0.1, AudioSourceKind.SYSTEM))  # overlap, summed
        out += m.feed(_frame(loud_mic, 0.1, 0.2, AudioSourceKind.MICROPHONE))  # emit chunk 0
        out += m.flush()
        # chunk 0 sums 0.8 + 0.1 = 0.9; dominant source = mic (higher energy)
        assert out[0].source is AudioSourceKind.MICROPHONE
        assert np.allclose(out[0].pcm, 0.9, atol=1e-5)

    def test_non_overlapping_sources_keep_separate_tags(self):
        m = MultiSourceMixer(chunk_seconds=0.1)
        mic = np.full(1600, 0.5, dtype=np.float32)
        sys = np.full(1600, 0.5, dtype=np.float32)
        out: list = []
        out += m.feed(_frame(mic, 0.0, 0.1, AudioSourceKind.MICROPHONE))
        out += m.feed(_frame(sys, 0.1, 0.2, AudioSourceKind.SYSTEM))
        out += m.feed(_frame(mic, 0.2, 0.3, AudioSourceKind.MICROPHONE))  # bump max_idx
        out += m.flush()
        tags = [f.source for f in out]
        assert tags[0] is AudioSourceKind.MICROPHONE
        assert tags[1] is AudioSourceKind.SYSTEM

    def test_clipping_to_unit_range(self):
        m = MultiSourceMixer(chunk_seconds=0.1)
        a = np.full(1600, 0.9, dtype=np.float32)
        b = np.full(1600, 0.9, dtype=np.float32)
        out: list = []
        out += m.feed(_frame(a, 0.0, 0.1, AudioSourceKind.MICROPHONE))
        out += m.feed(_frame(b, 0.0, 0.1, AudioSourceKind.SYSTEM))
        out += m.feed(_frame(a, 0.1, 0.2, AudioSourceKind.MICROPHONE))
        out += m.flush()
        # chunk 0 = 0.9+0.9 = 1.8, clipped to 1.0
        assert float(out[0].pcm.max()) <= 1.0
