"""Contract test: SpeakerDiarizer (task T015; contracts/speaker_diarizer.md).

Validates the protocol guarantees (stable arrival-order labels, reset(), ~80 ms
DiarFrames, max_speakers) against a conforming backend. The fake
``ManifestDiarizer`` is the offline reference; the real ``SortformerCoreMLDiarizer``
is smoke-checked under ``needs_models`` (skipped when the model isn't cached).
"""

from __future__ import annotations

import numpy as np
import pytest

from meeting_asr.diarization.sortformer_coreml import SortformerCoreMLDiarizer
from meeting_asr.types import AudioFrame, AudioSourceKind, ComputeBackend
from tests._fakes import ManifestDiarizer


def _timeline_frames(duration_s: float, step: float = 0.1):
    t = 0.0
    while t < duration_s:
        yield AudioFrame(pcm=np.zeros(int(step * 16000), dtype=np.float32), t_start=t, t_end=t + step,
                         source=AudioSourceKind.MICROPHONE)
        t += step


def test_diarizer_stable_arrival_order_labels(synthetic_fixture):
    manifest = synthetic_fixture("two_speaker_en")
    dia = ManifestDiarizer(manifest["turns"], latency_s=0.0)
    dia.load(ComputeBackend.CPU)
    dia.reset()
    collected = []
    for frame in _timeline_frames(11.0):
        collected.extend(dia.push(frame))
    speakers_seen = []
    for f in collected:
        if f.speaker_label not in speakers_seen:
            speakers_seen.append(f.speaker_label)
    assert speakers_seen == ["Speaker 1", "Speaker 2"]


def test_diarizer_emits_80ms_frames(synthetic_fixture):
    manifest = synthetic_fixture("single_speaker_en")
    dia = ManifestDiarizer(manifest["turns"], latency_s=0.0, frame_s=0.08)
    dia.reset()
    frames = []
    for fr in _timeline_frames(7.0):
        frames.extend(dia.push(fr))
    assert frames, "expected diar frames"
    widths = {round(f.t_end - f.t_start, 3) for f in frames}
    assert widths == {0.08}


def test_diarizer_reset_clears_state(synthetic_fixture):
    manifest = synthetic_fixture("single_speaker_en")
    dia = ManifestDiarizer(manifest["turns"], latency_s=0.0)
    for fr in _timeline_frames(7.0):
        dia.push(fr)
    assert dia._next_idx > 0
    dia.reset()
    assert dia._next_idx == 0


def test_diarizer_max_speakers_is_four():
    dia = ManifestDiarizer([{"speaker": "Speaker 1", "t_start": 0.0, "t_end": 1.0, "text": "x", "language": "en"}])
    assert dia.max_speakers() == 4


@pytest.mark.needs_models
def test_real_sortformer_smoke():
    dia = SortformerCoreMLDiarizer()
    try:
        dia.load(ComputeBackend.CPU)
    except Exception as e:
        pytest.skip(f"Sortformer model not cached: {e}")
    assert dia.max_speakers() == 4
