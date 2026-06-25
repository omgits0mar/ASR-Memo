"""US1 integration: the bridge drives a fake-but-real-pipeline live session end-to-end.

Uses the REAL ``meeting_asr.start_session`` + ``Pipeline`` with injected fakes
(``FixtureCapture`` + ``ManifestDiarizer``/``ManifestTranscriber``) over a synthetic
two-speaker fixture, asserting ordered ``segment`` events and a final ``status=stopped``
through the JS bridge — no window, no models, no network.
"""

from __future__ import annotations

import time

from app.bridge import Api
from meeting_asr import Backends, start_session
from meeting_asr.types import SystemReadinessReport
from tests._fakes import FixtureCapture, ManifestDiarizer, ManifestTranscriber


def _events(sink):
    import json
    import re

    out = []
    rx = re.compile(r"window\.onBackendEvent\((.*)\)$", re.DOTALL)
    for js in sink:
        m = rx.search(js.strip())
        if m:
            out.append(json.loads(m.group(1)))
    return out


def _drain_segments(sink, *, want=2, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        evs = [e for e in _events(sink) if e["type"] == "segment"]
        if len(evs) >= want:
            return evs
        time.sleep(0.02)
    return [e for e in _events(sink) if e["type"] == "segment"]


def _pipeline_start_factory(synthetic):
    """start_session wired to the fake backends over the synthetic fixture."""
    wav = synthetic["wav"]
    turns = synthetic["turns"]
    backends = Backends(
        capture=FixtureCapture(wav),
        diarizer=ManifestDiarizer(turns),
        transcriber=ManifestTranscriber(turns),
    )

    def _start(**kw):
        return start_session(_backends=backends, **kw)

    return _start


def test_bridge_live_session_end_to_end(synthetic_fixture):
    data = synthetic_fixture("two_speaker_en")
    sink = []
    api = Api(
        emit=sink.append,
        check_readiness_fn=lambda: SystemReadinessReport(
            models=[], mic_permission=True, system_audio_permission=True,
            compute_backend="cpu", os_supports_process_tap=True, missing=[],
        ),
        start_session_fn=_pipeline_start_factory(data),
    )

    res = api.start_live(["microphone"])
    assert "app_session_id" in res

    segs = _drain_segments(sink, want=3, timeout=6.0)
    # Wait for capture to finish, then stop.
    time.sleep(0.6)
    stop_res = api.stop_session()
    assert stop_res == {"status": "stopped"}

    # Status events drain through the async emit pump; wait for `stopped`.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status_evts = [e for e in _events(sink) if e["type"] == "status"]
        if any(e["status"] == "stopped" for e in status_evts):
            break
        time.sleep(0.02)

    all_evts = _events(sink)
    status_evts = [e for e in all_evts if e["type"] == "status"]
    # recording observed while active, stopped after stop_session.
    assert any(e["status"] == "recording" for e in status_evts)
    assert any(e["status"] == "stopped" for e in status_evts)

    # Segments are delivered in chronological start order (FR-019).
    starts = [e["segment"]["start"] for e in segs]
    assert starts == sorted(starts)

    # At least two distinct stable speaker labels appeared.
    labels = {e["segment"]["speaker_label"] for e in segs}
    assert len(labels) >= 2

    # The final transcript snapshot contains every delivered segment.
    snap = api.get_transcript()
    assert len(snap["segments"]) >= 2
    assert len(snap["speakers"]) >= 2
