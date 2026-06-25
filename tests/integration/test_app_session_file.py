"""US2 integration: the bridge drives a file-import run end-to-end (task T025).

Uses the REAL ``meeting_asr.transcribe_file`` + ``Pipeline`` with injected fakes
(``FileCapture`` + ``ManifestDiarizer``/``ManifestTranscriber``) over a synthetic
fixture, asserting ordered ``segment`` events, a ``progress``→1.0 sequence, and a
final ``status=stopped`` through the JS bridge — no window, no models, no network.
"""

from __future__ import annotations

import json
import re
import time

import pytest

from app.bridge import Api
from meeting_asr import Backends, transcribe_file
from meeting_asr.audio.file_capture import FileCapture
from meeting_asr.types import SystemReadinessReport
from tests._fakes import ManifestDiarizer, ManifestTranscriber


def _events(sink):
    out = []
    rx = re.compile(r"window\.onBackendEvent\((.*)\)$", re.DOTALL)
    for js in sink:
        m = rx.search(js.strip())
        if m:
            out.append(json.loads(m.group(1)))
    return out


def test_bridge_file_import_end_to_end(synthetic_fixture):
    data = synthetic_fixture("two_speaker_en")
    wav, turns = data["wav"], data["turns"]
    backends = Backends(
        capture=FileCapture(wav),
        diarizer=ManifestDiarizer(turns),
        transcriber=ManifestTranscriber(turns),
    )

    sink = []
    api = Api(
        emit=sink.append,
        check_readiness_fn=lambda: SystemReadinessReport(
            models=[], mic_permission=True, system_audio_permission=True,
            compute_backend="cpu", os_supports_process_tap=True, missing=[],
        ),
        transcribe_file_fn=lambda **kw: transcribe_file(_backends=backends, **kw),
    )

    res = api.transcribe_file(wav)
    assert "app_session_id" in res

    # Wait for the run to reach a terminal status.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        evs = _events(sink)
        if any(e.get("type") == "status" and e.get("status") in ("stopped", "error") for e in evs):
            break
        time.sleep(0.05)

    evs = _events(sink)
    segs = [e for e in evs if e["type"] == "segment"]
    assert len(segs) >= 2
    # Chronological start order (FR-019).
    starts = [e["segment"]["start"] for e in segs]
    assert starts == sorted(starts)
    # Progress advanced monotonically to 1.0.
    fracs = [e["fraction"] for e in evs if e["type"] == "progress"]
    assert fracs and fracs[-1] == pytest.approx(1.0)
    assert fracs == sorted(fracs)
    # Terminal status stopped (not error) for a good file.
    assert any(e["type"] == "status" and e["status"] == "stopped" for e in evs)
