"""Integration test: US3 multilingual per-segment language (task T033; SC-003).

End-to-end: multilingual fixture (en/es alternating across speakers) → pipeline →
asserts per-segment ``language`` tags match the spoken language, two languages
coexist, and a ``language_hint`` biases without forcing a single language.
"""

from __future__ import annotations

from meeting_asr import AudioSourceKind, Backends, start_session
from tests._fakes import FixtureCapture, ManifestDiarizer, ManifestTranscriber
from tests.integration.test_us1_single_source import _drain_until_stable


def _backends(manifest):
    return Backends(
        capture=FixtureCapture(manifest["wav"], source=AudioSourceKind.MICROPHONE),
        diarizer=ManifestDiarizer(manifest["turns"]),
        transcriber=ManifestTranscriber(manifest["turns"]),
    )


def test_us3_per_segment_language_matches_spoken_language(synthetic_fixture):
    manifest = synthetic_fixture("multilingual")
    session = start_session(_backends=_backends(manifest))
    try:
        _drain_until_stable(session)
    finally:
        final = session.stop(timeout_s=5.0)

    assert final, "expected multilingual segments"
    # Build the reference: for each session-time window, the manifest's turn language.
    turns = sorted(manifest["turns"], key=lambda t: t["t_start"])
    matched, total = 0, 0
    for seg in final:
        # find the manifest turn covering this segment's midpoint
        mid = (seg.start + seg.end) / 2
        ref = next((t for t in turns if t["t_start"] <= mid <= t["t_end"]), None)
        if ref is None:
            continue
        total += 1
        if seg.language == ref["language"]:
            matched += 1
    assert total > 0
    # SC-003: ≥95% per-segment language-ID accuracy (100% with the manifest-driven fake;
    # the real Nemotron path is gated under needs_models).
    assert matched / total >= 0.95


def test_us3_two_languages_coexist(synthetic_fixture):
    manifest = synthetic_fixture("multilingual")
    session = start_session(_backends=_backends(manifest))
    try:
        _drain_until_stable(session)
    finally:
        final = session.stop(timeout_s=5.0)
    langs = {s.language for s in final}
    assert "en" in langs and "es" in langs


def test_us3_language_hint_biases_without_forcing(synthetic_fixture):
    manifest = synthetic_fixture("multilingual")
    # Pass a hint; the fake honors per-token language regardless, so per-segment tags survive.
    session = start_session(_backends=_backends(manifest), language_hint="en")
    try:
        _drain_until_stable(session)
    finally:
        final = session.stop(timeout_s=5.0)
    langs = {s.language for s in final}
    assert "es" in langs  # hint did not force everything into 'en'
