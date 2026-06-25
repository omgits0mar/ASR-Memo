"""Integration test: US2 merged mic + system audio (task T026; FR-009, FR-015, SC-004).

Two sources (MIC + SYSTEM) merged onto one timeline via CompositeCapture +
MultiSourceMixer, with ≥3 distinct speakers, overlap handling, and the
system-audio permission-denied error path.
"""

from __future__ import annotations

import pytest

from meeting_asr import AudioSourceKind, Backends, ReadinessError, start_session
from meeting_asr.audio.coreaudio_tap import CoreAudioTapCapture
from meeting_asr.audio.mixer import CompositeCapture
from meeting_asr._logging import CapturePermissionError
from tests._fakes import FixtureCapture, ManifestDiarizer, ManifestTranscriber
from tests.integration.test_us1_single_source import _drain_until_stable


def _merged_backends(manifest, mic_wav, sys_wav):
    capture = CompositeCapture([
        FixtureCapture(mic_wav, source=AudioSourceKind.MICROPHONE),
        FixtureCapture(sys_wav, source=AudioSourceKind.SYSTEM),
    ])
    return Backends(
        capture=capture,
        diarizer=ManifestDiarizer(manifest),
        transcriber=ManifestTranscriber(manifest),
    )


def test_us2_merged_timeline_three_speakers(synthetic_fixture):
    base = synthetic_fixture("two_speaker_en")
    # Combined manifest: mic speakers (1,2) + a remote Speaker 3 turn overlapping.
    combined = list(base["turns"]) + [
        {"speaker": "Speaker 3", "t_start": 3.0, "t_end": 4.6,
         "text": "remote participant joins the call", "language": "en"},
    ]
    session = start_session(_backends=_merged_backends(combined, base["wav"], base["wav"]))
    try:
        _drain_until_stable(session)
    finally:
        final = session.stop(timeout_s=5.0)

    labels = {s.speaker_label for s in final}
    assert {"Speaker 1", "Speaker 2", "Speaker 3"} <= labels
    # merged transcript is chronological
    starts = [s.start for s in final]
    assert starts == sorted(starts)
    # the remote participant's words all appear in the merged transcript (best-effort
    # attribution under overlap may interleave, so check per-word presence)
    joined = " ".join(s.text for s in final).lower()
    for word in ("remote", "participant", "joins", "call"):
        assert word in joined


def test_us2_overlap_keeps_both_speakers(synthetic_fixture):
    base = synthetic_fixture("two_speaker_en")
    # Speaker 3 overlaps Speaker 1's first turn (0.2–1.8).
    combined = list(base["turns"]) + [
        {"speaker": "Speaker 3", "t_start": 0.5, "t_end": 1.6,
         "text": "talking over each other", "language": "en"},
    ]
    session = start_session(_backends=_merged_backends(combined, base["wav"], base["wav"]))
    try:
        _drain_until_stable(session)
    finally:
        final = session.stop(timeout_s=5.0)
    labels = {s.speaker_label for s in final}
    assert "Speaker 1" in labels and "Speaker 3" in labels


def test_us2_system_permission_denied_raises(monkeypatch):
    from meeting_asr.models import readiness

    monkeypatch.setattr(readiness, "os_supports_process_tap", lambda: False)
    with pytest.raises(ReadinessError):
        start_session(sources=(AudioSourceKind.MICROPHONE, AudioSourceKind.SYSTEM))


def test_us2_coreaudio_tap_permission_denied_on_unsupported_os(monkeypatch):
    from meeting_asr.audio import coreaudio_tap

    monkeypatch.setattr(coreaudio_tap, "os_supports_process_tap", lambda: False)
    cap = CoreAudioTapCapture(helper_path="/usr/bin/false")
    with pytest.raises(CapturePermissionError):
        cap.start(lambda _f: None)
