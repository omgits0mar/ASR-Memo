"""Unit tests for FileCapture — file → 16 kHz mono frames (task T023 / US2).

Promotes the test ``FixtureCapture`` mechanism to a production ``AudioCapture``:
decode + downmix + resample, continuous session-clock frames, progress fraction,
idempotent stop, and structured ``audio.unreadable`` / ``audio.empty`` errors.
All offline (synthetic WAVs written to tmp_path).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import soundfile as sf

from meeting_asr.audio.file_capture import FileCapture, FileCaptureError
from meeting_asr.audio.mixer import SAMPLE_RATE
from meeting_asr.types import AudioSourceKind, CaptureState


def _write_wav(path, pcm, rate):
    sf.write(str(path), np.asarray(pcm, dtype=np.float32), rate)


def _collect(capture, *, timeout=5.0):
    frames = []
    capture.start(frames.append)
    deadline = time.monotonic() + timeout
    while capture.consumed_fraction() < 1.0 and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)  # let the final frame(s) land
    capture.stop()
    return frames


def test_decodes_wav_and_resamples_to_16k_mono(tmp_path):
    # 8 kHz mono tone, 2 s → must be resampled to 16 kHz mono on read.
    sr_in = 8000
    t = np.linspace(0, 2.0, sr_in * 2, endpoint=False)
    pcm = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    wav = tmp_path / "in.wav"
    _write_wav(wav, pcm, sr_in)

    cap = FileCapture(str(wav))
    assert cap.kind is AudioSourceKind.MICROPHONE
    assert cap.permission_status() is True
    assert cap.total_seconds() == pytest.approx(2.0, abs=0.05)

    frames = _collect(cap)
    assert frames, "expected at least one frame"
    total_samples = sum(f.n_samples for f in frames)
    assert total_samples == pytest.approx(SAMPLE_RATE * 2.0, rel=0.05)
    # Every frame is 16 kHz mono float32 on a continuous clock.
    for f in frames:
        assert f.pcm.dtype == np.float32
        assert f.source is AudioSourceKind.MICROPHONE
    starts = [f.t_start for f in frames]
    assert starts == sorted(starts)
    assert frames[0].t_start == pytest.approx(0.0, abs=1e-6)
    # Continuous: each frame's start ≈ previous end.
    for a, b in zip(frames, frames[1:]):
        assert b.t_start == pytest.approx(a.t_end, abs=1e-3)


def test_downmixes_stereo_to_mono(tmp_path):
    sr = 16000
    n = sr  # 1 s
    stereo = np.stack([np.full(n, 0.4, np.float32), np.full(n, 0.2, np.float32)], axis=1)
    wav = tmp_path / "stereo.wav"
    _write_wav(wav, stereo, sr)
    cap = FileCapture(str(wav))
    frames = _collect(cap)
    pcm = np.concatenate([f.pcm for f in frames])
    assert pcm.ndim == 1
    # mean of [0.4, 0.2] = 0.3
    assert float(np.mean(np.abs(pcm))) == pytest.approx(0.3, abs=0.02)


def test_consumed_fraction_is_monotonic_to_one(tmp_path):
    sr = 16000
    wav = tmp_path / "tone.wav"
    _write_wav(wav, 0.1 * np.sin(2 * np.pi * 300 * np.arange(sr) / sr).astype(np.float32), sr)
    cap = FileCapture(str(wav), block_seconds=0.1)
    fracs = []

    def on_frame(_):
        fracs.append(cap.consumed_fraction())

    cap.start(on_frame)
    while cap.consumed_fraction() < 1.0:
        time.sleep(0.005)
    cap.stop()
    assert fracs == sorted(fracs)          # monotonic non-decreasing
    assert fracs[-1] == pytest.approx(1.0)  # reaches 1.0 at completion
    assert all(0.0 <= f <= 1.0 for f in fracs)


def test_stop_is_idempotent(tmp_path):
    sr = 16000
    wav = tmp_path / "tone.wav"
    _write_wav(wav, np.zeros(sr, np.float32), sr)
    cap = FileCapture(str(wav))
    cap.start(lambda _f: None)
    time.sleep(0.05)
    cap.stop()
    cap.stop()  # must not raise
    assert cap.state() in (CaptureState.IDLE, CaptureState.ERROR)


def test_missing_file_raises_unreadable(tmp_path):
    with pytest.raises(FileCaptureError) as ei:
        FileCapture(str(tmp_path / "nope.wav"))
    assert ei.value.info.code == "audio.unreadable"


def test_unsupported_format_raises_unreadable(tmp_path):
    bad = tmp_path / "notaudio.bin"
    bad.write_bytes(b"\x00\x01\x02 not actually audio")
    with pytest.raises(FileCaptureError) as ei:
        FileCapture(str(bad))
    assert ei.value.info.code == "audio.unreadable"


def test_empty_audio_raises_empty(tmp_path):
    wav = tmp_path / "empty.wav"
    _write_wav(wav, np.zeros(0, np.float32), 16000)
    with pytest.raises(FileCaptureError) as ei:
        FileCapture(str(wav))
    assert ei.value.info.code == "audio.empty"


def test_silence_emits_frames_no_fabrication(tmp_path):
    """Silent input yields frames (zeros) but the capture invents nothing (edge T050)."""
    sr = 16000
    wav = tmp_path / "silence.wav"
    _write_wav(wav, np.zeros(int(sr * 0.5), np.float32), sr)
    cap = FileCapture(str(wav))
    frames = _collect(cap)
    assert frames
    pcm = np.concatenate([f.pcm for f in frames])
    assert float(np.max(np.abs(pcm))) == pytest.approx(0.0, abs=1e-6)
