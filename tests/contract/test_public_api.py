"""Contract test: public in-process API (task T017; contracts/public_api.md).

Validates ``start_session`` / ``segments`` / ``transcript`` / ``stop``, structured
dataclass results, the sequential-session ``SessionBusyError`` guard, and the
``ReadinessError`` / mic-permission paths — against injected fakes via ``Backends``.
"""

from __future__ import annotations

import time

import pytest

import meeting_asr
from meeting_asr import (
    AudioSourceKind,
    Backends,
    ReadinessError,
    SessionBusyError,
    TranscriptSegment,
    start_session,
)
from meeting_asr.types import SessionStatus
from tests._fakes import FixtureCapture, ManifestDiarizer, ManifestTranscriber


def _backends_for(manifest, wav):
    return Backends(
        capture=FixtureCapture(wav, source=AudioSourceKind.MICROPHONE),
        diarizer=ManifestDiarizer(manifest["turns"]),
        transcriber=ManifestTranscriber(manifest["turns"]),
    )


def test_start_session_returns_active_session_with_id(synthetic_fixture):
    manifest = synthetic_fixture("single_speaker_en")
    session = start_session(_backends=_backends_for(manifest, manifest["wav"]))
    try:
        assert session.session_id
        assert session.status == SessionStatus.ACTIVE
        assert isinstance(session.sources, list) and session.sources[0].kind is AudioSourceKind.MICROPHONE
    finally:
        session.stop(timeout_s=5.0)


def test_stop_returns_final_transcript_and_sets_stopped(synthetic_fixture):
    manifest = synthetic_fixture("two_speaker_en")
    session = start_session(_backends=_backends_for(manifest, manifest["wav"]))
    time.sleep(0.5)  # let the fixture stream + workers drain
    final = session.stop(timeout_s=5.0)
    assert session.status == SessionStatus.STOPPED
    assert isinstance(final, list)
    assert final and all(isinstance(s, TranscriptSegment) for s in final)


def test_transcript_snapshot_is_chronological_during_session(synthetic_fixture):
    manifest = synthetic_fixture("two_speaker_en")
    session = start_session(_backends=_backends_for(manifest, manifest["wav"]))
    try:
        time.sleep(0.5)
        snap = session.transcript()
        starts = [s.start for s in snap]
        assert starts == sorted(starts)
        assert all(isinstance(s, TranscriptSegment) for s in snap)
    finally:
        session.stop(timeout_s=5.0)


def test_on_segment_callback_fires(synthetic_fixture):
    manifest = synthetic_fixture("two_speaker_en")
    received = []
    session = start_session(
        _backends=_backends_for(manifest, manifest["wav"]),
        on_segment=received.append,
    )
    try:
        time.sleep(0.6)
    finally:
        session.stop(timeout_s=5.0)
    assert received, "on_segment should have fired for finalized segments"
    assert all(isinstance(s, TranscriptSegment) for s in received)


def test_sequential_session_busy_guard(synthetic_fixture):
    manifest = synthetic_fixture("single_speaker_en")
    first = start_session(_backends=_backends_for(manifest, manifest["wav"]))
    try:
        with pytest.raises(SessionBusyError):
            start_session(_backends=_backends_for(manifest, manifest["wav"]))
    finally:
        first.stop(timeout_s=5.0)
    # after stop, a new session is allowed
    second = start_session(_backends=_backends_for(manifest, manifest["wav"]))
    second.stop(timeout_s=5.0)


def test_segments_blocking_iterator_yields(synthetic_fixture):
    import threading

    manifest = synthetic_fixture("single_speaker_en")
    session = start_session(_backends=_backends_for(manifest, manifest["wav"]))
    yielded = []

    def _drain():
        for seg in session.segments():  # blocks until the session ends
            yielded.append(seg)

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    try:
        # Let the fixture stream + workers produce segments, then end the stream.
        time.sleep(0.6)
    finally:
        session.stop(timeout_s=5.0)
    t.join(timeout=5.0)
    assert yielded, "iterator should have yielded finalized segments"
    assert isinstance(yielded[0], TranscriptSegment)


def test_default_path_without_models_raises_readiness():
    # No _backends → real backends → models not cached → ReadinessError (not a crash)
    with pytest.raises(ReadinessError):
        start_session()


def test_system_source_on_unsupported_os_raises_readiness(monkeypatch):
    from meeting_asr.models import readiness

    monkeypatch.setattr(readiness, "os_supports_process_tap", lambda: False)
    with pytest.raises(ReadinessError):
        start_session(sources=(AudioSourceKind.MICROPHONE, AudioSourceKind.SYSTEM))
