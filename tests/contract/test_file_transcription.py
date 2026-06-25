"""Contract test for transcribe_file (task T024 / US2).

Validates the facade entry point with injected fakes (offline): ordered transcript,
on_progress → 1.0, not_ready / session.busy rules, and an ERROR session for a bad
path. Mirrors tests/contract/test_lifecycle.py's seam-driven style.
"""

from __future__ import annotations

import pytest

from meeting_asr import Backends, start_session, transcribe_file
from meeting_asr.audio.file_capture import FileCapture
from meeting_asr.types import AudioSourceKind, SessionStatus
from tests._fakes import FixtureCapture, ManifestDiarizer, ManifestTranscriber


def _fake_backends(synthetic):
    wav = synthetic["wav"]
    turns = synthetic["turns"]
    return Backends(
        capture=FileCapture(wav),
        diarizer=ManifestDiarizer(turns),
        transcriber=ManifestTranscriber(turns),
    )


def test_transcribe_file_ordered_transcript_and_progress(synthetic_fixture):
    data = synthetic_fixture("single_speaker_en")
    progress = []
    segs = []
    session = transcribe_file(
        data["wav"],
        _backends=_fake_backends(data),
        on_segment=lambda s: segs.append(s),
        on_progress=lambda f: progress.append(f),
    )
    assert session.status is SessionStatus.STOPPED
    # Ordered transcript (chronological start).
    starts = [s.start for s in segs]
    assert starts == sorted(starts)
    assert len(session.transcript()) >= 1
    # Progress reached 1.0.
    assert progress and progress[-1] == pytest.approx(1.0)
    assert progress == sorted(progress)  # monotonic


def test_transcribe_file_bad_path_returns_error_session(tmp_path):
    errs = []
    session = transcribe_file(
        str(tmp_path / "missing.wav"),
        on_error=lambda e: errs.append(e),
    )
    assert session.status is SessionStatus.ERROR
    assert session.error is not None
    assert session.error.code == "audio.unreadable"
    assert errs and errs[0].code == "audio.unreadable"


def test_transcribe_file_not_ready_without_models(tmp_path):
    import soundfile as sf
    import numpy as np

    wav = tmp_path / "t.wav"
    sf.write(str(wav), np.zeros(16000, np.float32), 16000)
    # Default backends need cached models → ReadinessError (offline, no models).
    with pytest.raises(Exception):  # ReadinessError
        transcribe_file(str(wav))


def test_transcribe_file_busy_when_session_active(synthetic_fixture):
    data = synthetic_fixture("two_speaker_en")
    # Occupy the sequential-session guard with an active fake live session.
    active = start_session(
        sources=(AudioSourceKind.MICROPHONE,),
        _backends=Backends(
            capture=FixtureCapture(data["wav"]),
            diarizer=ManifestDiarizer(data["turns"]),
            transcriber=ManifestTranscriber(data["turns"]),
        ),
    )
    try:
        with pytest.raises(Exception):  # SessionBusyError
            transcribe_file(data["wav"], _backends=_fake_backends(data))
    finally:
        active.stop()
