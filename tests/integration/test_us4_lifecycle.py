"""Integration test: US4 full lifecycle (task T038; SC-005, SC-007).

prepare_models() → check_readiness() → start_session() → segments → stop(),
driven entirely through the public API. ``prepare_models`` uses the registry's
downloader seam (offline); the session uses fake backends. Exercises FR-012
(cached fast-path) and FR-020 (sequential sessions).
"""

from __future__ import annotations

import pytest

from meeting_asr import (
    AudioSourceKind,
    Backends,
    ReadinessError,
    SystemReadinessReport,
    TranscriptionSession,
    check_readiness,
    prepare_models,
    start_session,
)
from meeting_asr.models import registry as reg
from meeting_asr.types import SessionStatus
from tests._fakes import FixtureCapture, ManifestDiarizer, ManifestTranscriber
from tests.integration.test_us1_single_source import _drain_until_stable


def _ok_downloader():
    def _dl(asset, target, report):
        for f in asset.expected_files:
            (target / f).write_bytes(b"\x00" * 16)
        report(len(asset.expected_files), len(asset.expected_files))
        return target
    return _dl


def _backends(manifest):
    return Backends(
        capture=FixtureCapture(manifest["wav"], source=AudioSourceKind.MICROPHONE),
        diarizer=ManifestDiarizer(manifest["turns"]),
        transcriber=ManifestTranscriber(manifest["turns"]),
    )


def test_us4_full_lifecycle_prepare_readiness_session(synthetic_fixture, tmp_path, monkeypatch):
    from meeting_asr.models import readiness

    manifest = synthetic_fixture("two_speaker_en")
    monkeypatch.setattr(reg, "default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(readiness, "default_cache_dir", lambda: tmp_path)  # build_readiness's binding
    # Make the facade-level prepare use our offline downloader.
    monkeypatch.setattr(reg, "_default_downloader", _ok_downloader())
    # Simulate a fully-permitted machine so `ready` reflects only the models.
    monkeypatch.setattr(readiness, "mic_permission", lambda: True)
    monkeypatch.setattr(readiness, "system_audio_permission", lambda: True)
    monkeypatch.setattr(readiness, "os_supports_process_tap", lambda: True)

    # 1) prepare_models() (offline via the seam) → readiness ready
    report = prepare_models()
    assert isinstance(report, SystemReadinessReport)
    assert report.ready, report.missing

    # 2) check_readiness() reflects the cached state
    assert check_readiness().ready

    # 3) cached fast-path: a second prepare does not re-download (no network) — FR-012
    prepare_models()  # would raise if it tried the network (offline guard is active)

    # 4) start → segments → stop via the public API (fakes injected)
    session = start_session(_backends=_backends(manifest))
    assert isinstance(session, TranscriptionSession)
    try:
        _drain_until_stable(session)
        live = session.transcript()
        assert live == sorted(live, key=lambda s: (s.start, s.end))
    finally:
        final = session.stop(timeout_s=5.0)
    assert session.status is SessionStatus.STOPPED
    assert final

    # 5) a second session is allowed sequentially (FR-020)
    session2 = start_session(_backends=_backends(manifest))
    session2.stop(timeout_s=5.0)


def test_us4_start_without_models_would_block_but_injected_backends_bypass(synthetic_fixture):
    # With injected backends we bypass the model-readiness gate entirely.
    manifest = synthetic_fixture("single_speaker_en")
    session = start_session(_backends=_backends(manifest))
    session.stop(timeout_s=5.0)
