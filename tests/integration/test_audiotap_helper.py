"""Integration test: recorded tap dump → CoreAudioTapCapture (task T027).

The real Swift helper emits raw float32 PCM on stdout. Here a tiny fake helper
script replays a known float32 dump; CoreAudioTapCapture reads the pipe and must
produce 16 kHz mono float32 AudioFrames tagged source=SYSTEM.
"""

from __future__ import annotations

import os
import stat
import textwrap
import threading
import time

import numpy as np

from meeting_asr.audio.coreaudio_tap import CoreAudioTapCapture
from meeting_asr.types import AudioSourceKind, CaptureState


def _make_fake_helper(tmp_path, samples: np.ndarray) -> str:
    """Write a shell helper that dumps the given float32 samples to stdout."""
    dump = tmp_path / "tap_dump.pcm"
    samples.astype(np.float32).tofile(str(dump))
    helper = tmp_path / "fake_audiotap.sh"
    helper.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        cat "{dump}"
        """))
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(helper)


def test_audiotap_helper_produces_16k_mono_frames(tmp_path):
    samples = np.full(4800, 0.5, dtype=np.float32)  # 0.3 s @ 16 kHz of constant 0.5
    helper = _make_fake_helper(tmp_path, samples)

    cap = CoreAudioTapCapture(helper_path=helper, block_seconds=0.1)
    assert cap.kind is AudioSourceKind.SYSTEM
    assert cap.state() is CaptureState.IDLE

    frames = []
    done = threading.Event()

    def _on_frame(frame):
        frames.append(frame)
        if len(frames) >= 2:
            done.set()

    cap.start(_on_frame)
    try:
        assert done.wait(timeout=5.0), "no frames produced from the tap dump"
    finally:
        cap.stop()

    assert frames
    for f in frames:
        assert f.source is AudioSourceKind.SYSTEM
        assert f.pcm.dtype == np.float32
        # constant 0.5 input → ~0.5 output (no scaling beyond normalize)
        assert abs(float(np.mean(f.pcm)) - 0.5) < 0.1


def _make_fake_helper_with_stderr(tmp_path, samples: np.ndarray, stderr_lines: str) -> str:
    """Fake helper that prints `stderr_lines` to stderr, then dumps PCM to stdout."""
    dump = tmp_path / "tap_dump.pcm"
    samples.astype(np.float32).tofile(str(dump))
    helper = tmp_path / "fake_audiotap_stderr.sh"
    helper.write_text(textwrap.dedent(f"""\
        #!/bin/bash
        printf '{stderr_lines}' >&2
        cat "{dump}"
        """))
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(helper)


def test_audiotap_honors_announced_native_rate(tmp_path):
    # 0.2 s of constant 0.5 at the announced 48 kHz native rate (9600 samples).
    samples = np.full(9600, 0.5, dtype=np.float32)
    helper = _make_fake_helper_with_stderr(tmp_path, samples, "rate=48000\\nchannels=1\\n")

    cap = CoreAudioTapCapture(helper_path=helper, block_seconds=0.1)
    frames = []
    done = threading.Event()

    def _on_frame(frame):
        frames.append(frame)
        if len(frames) >= 1:
            done.set()

    cap.start(_on_frame)
    try:
        assert done.wait(timeout=5.0), "no frames produced"
        # Let the stderr drain thread pick up `rate=48000`.
        time.sleep(0.2)
        assert cap._native_rate == 48000
    finally:
        cap.stop()

    # Each stdout read is `block_seconds` worth of samples at 16 kHz (1600), but
    # those are 48 kHz input samples → resampled down by ~3× to ~533 output.
    # The point: the helper's announced 48 kHz rate drives the resample (not the
    # old 16 kHz assumption, which would pass input through unchanged).
    assert frames
    per_frame = frames[0].pcm.size
    assert 400 <= per_frame <= 700, f"expected ~3× downsample (≈533), got {per_frame}"


def test_audiotap_permission_denial_sets_state(tmp_path):
    helper = tmp_path / "deny.sh"
    helper.write_text(textwrap.dedent("""\
        #!/bin/bash
        echo "permission denied: audio-capture authorization timed out" >&2
        exit 2
        """))
    helper.chmod(helper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    cap = CoreAudioTapCapture(helper_path=str(helper), block_seconds=0.1)
    cap.start(lambda _f: None)
    # Read loop ends immediately (helper exits 2); state must reflect the denial.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and cap.state() is not CaptureState.PERMISSION_DENIED:
        time.sleep(0.02)
    cap.stop()
    assert cap.state() is CaptureState.PERMISSION_DENIED


def test_audiotap_missing_helper_raises(tmp_path):
    from meeting_asr._logging import CaptureDeviceError

    cap = CoreAudioTapCapture(helper_path=str(tmp_path / "does_not_exist"))
    # _default_helper_path resolves None when the real helper is absent, but we set
    # an explicit missing path → start attempts Popen and surfaces a device error.
    import pytest

    with pytest.raises((CaptureDeviceError, FileNotFoundError, OSError)):
        cap.start(lambda _f: None)
