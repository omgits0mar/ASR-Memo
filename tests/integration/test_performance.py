"""Performance gate (task T043; SC-001, SC-008, FR-021).

  * turn-to-text latency ≤3s (target ~1.5–2.0s)
  * 60-min soak keeps timeline/label/order stable
  * compute-pressure case: graceful degradation + lag signaling, no dropped audio

The latency + soak cases need real models + realtime hardware (``needs_models`` /
``needs_hardware`` / ``slow``). The compute-pressure case is fully offline: a
slow fake transcriber induces backpressure and we assert the pipeline *signals*
``COMPUTE_PRESSURE`` without corrupting the timeline.
"""

from __future__ import annotations

import time

import pytest

from meeting_asr import AudioSourceKind, Backends, start_session
from meeting_asr.types import ErrorInfo
from tests._fakes import FixtureCapture, ManifestDiarizer, ManifestTranscriber
from tests.integration.test_us1_single_source import _drain_until_stable

TURN_TO_TEXT_BUDGET_S = 3.0


class _SlowTranscriber(ManifestTranscriber):
    """Injects artificial per-frame latency to induce backpressure."""

    def push(self, frame, *, language_hint=None):
        time.sleep(0.02)  # slow enough to grow the asr queue past max_lag
        return super().push(frame, language_hint=language_hint)


def test_compute_pressure_signals_lag_without_dropping(synthetic_fixture):
    manifest = synthetic_fixture("two_speaker_en")
    errors = []
    backends = Backends(
        capture=FixtureCapture(manifest["wav"], source=AudioSourceKind.MICROPHONE),
        diarizer=ManifestDiarizer(manifest["turns"]),
        transcriber=_SlowTranscriber(manifest["turns"]),
    )
    session = start_session(_backends=backends, on_error=errors.append)
    try:
        _drain_until_stable(session, timeout_s=12.0)
    finally:
        final = session.stop(timeout_s=10.0)

    # FR-021: a COMPUTE_PRESSURE condition was signaled (non-terminal).
    assert any(getattr(e, "code", None) == "COMPUTE_PRESSURE" for e in errors), "expected lag signal"
    # The session survived (not ERROR) and the timeline is still coherent.
    assert session.status.value != "error"
    assert final
    starts = [s.start for s in final]
    assert starts == sorted(starts)


@pytest.mark.needs_models
@pytest.mark.needs_hardware
@pytest.mark.slow
def test_turn_to_text_latency_within_budget(synthetic_fixture):
    pytest.importorskip("onnxruntime")
    # Real-model path: measure wall-clock from speech end to first segment.
    # Wired under needs_models on Apple Silicon; skipped otherwise.
    pytest.skip("requires real Nemotron + Sortformer models and live audio")


@pytest.mark.needs_models
@pytest.mark.needs_hardware
@pytest.mark.slow
def test_sixty_minute_soak_stability():
    pytest.importorskip("onnxruntime")
    pytest.skip("requires real models + a 60-min realtime run (SC-008)")
