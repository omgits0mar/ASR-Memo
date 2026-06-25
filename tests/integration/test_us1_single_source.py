"""Integration test: US1 single-source live diarized transcription (task T018).

End-to-end: mic fixture → (mix) → diarize ∥ transcribe → fuse → session, via the
real public ``start_session`` with fake backends replaying a known manifest.

Asserts the US1 acceptance scenarios:
  * streamed TranscriptSegments carry speaker label + start/end + text
  * a second voice gets a distinct, stable "Speaker 2" label
  * transcript() returns chronological segments
  * stop() returns the final transcript
"""

from __future__ import annotations

import time

import pytest

from meeting_asr import AudioSourceKind, Backends, TranscriptSegment, start_session
from meeting_asr.types import SessionStatus
from tests._fakes import FixtureCapture, ManifestDiarizer, ManifestTranscriber


def _drain_until_stable(session, *, timeout_s: float = 8.0, stable_s: float = 0.3) -> list:
    """Poll transcript() until its length stops growing, or timeout."""
    deadline = time.monotonic() + timeout_s
    last_len, last_change = -1, time.monotonic()
    while time.monotonic() < deadline:
        snap = session.transcript()
        if len(snap) != last_len:
            last_len, last_change = len(snap), time.monotonic()
        elif time.monotonic() - last_change > stable_s:
            return snap
        time.sleep(0.05)
    return session.transcript()


def _backends(manifest):
    return Backends(
        capture=FixtureCapture(manifest["wav"], source=AudioSourceKind.MICROPHONE),
        diarizer=ManifestDiarizer(manifest["turns"]),
        transcriber=ManifestTranscriber(manifest["turns"]),
    )


def test_us1_streamed_segments_carry_label_time_text(synthetic_fixture):
    manifest = synthetic_fixture("two_speaker_en")
    streamed = []
    session = start_session(_backends=_backends(manifest), on_segment=streamed.append)
    try:
        _drain_until_stable(session)
    finally:
        final = session.stop(timeout_s=5.0)

    all_segs = streamed or final
    assert all_segs, "no segments were produced"
    for s in all_segs:
        assert isinstance(s, TranscriptSegment)
        assert s.speaker_label  # label present
        assert s.end >= s.start  # timestamps coherent
        assert s.text  # recognized text present


def test_us1_second_voice_gets_speaker_2(synthetic_fixture):
    manifest = synthetic_fixture("two_speaker_en")
    session = start_session(_backends=_backends(manifest))
    try:
        _drain_until_stable(session)
    finally:
        final = session.stop(timeout_s=5.0)

    labels = [s.speaker_label for s in final]
    assert "Speaker 1" in labels and "Speaker 2" in labels
    # arrival order: Speaker 1 appears before Speaker 2
    assert labels.index("Speaker 1") < labels.index("Speaker 2")


def test_us1_transcript_is_chronological_and_complete(synthetic_fixture):
    manifest = synthetic_fixture("two_speaker_en")
    session = start_session(_backends=_backends(manifest))
    try:
        _drain_until_stable(session)
        live = session.transcript()
        # live snapshot is chronological (FR-019)
        starts = [s.start for s in live]
        assert starts == sorted(starts)
    finally:
        final = session.stop(timeout_s=5.0)

    assert session.status == SessionStatus.STOPPED
    assert final, "stop() must return the final transcript"
    # the reconstructed transcript covers the manifest's words
    joined = " ".join(s.text for s in final).lower()
    for word in ("morning", "roadmap", "priorities"):
        assert word in joined, f"expected transcript word '{word}' missing"


def test_us1_speakers_roster_stable(synthetic_fixture):
    manifest = synthetic_fixture("two_speaker_en")
    session = start_session(_backends=_backends(manifest))
    try:
        _drain_until_stable(session)
        roster = session.speakers()
        assert set(roster) <= {"Speaker 1", "Speaker 2"}
        for label, spk in roster.items():
            assert spk.label == label
            assert spk.last_seen >= spk.first_seen
    finally:
        session.stop(timeout_s=5.0)
