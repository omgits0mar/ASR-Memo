"""Headless contract tests for the JS↔Python webview bridge (task T015 + US1/US3).

Instantiates :class:`app.bridge.Api` with injected backend fakes and a stub
``evaluate_js`` sink (no WKWebView window), then asserts method return-shapes,
event sequence/ordering, and the not-ready / session-busy rejections — all offline.

Shared by the US1–US4 bridge contract assertions.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import List

import pytest

from meeting_asr.types import (
    AudioSourceKind,
    ErrorInfo,
    SessionStatus,
    Speaker,
    SystemReadinessReport,
    TranscriptSegment,
)

# Imported lazily inside tests so a missing app package surfaces a clear error.


# --------------------------------------------------------------------------- #
# Shared harness helpers
# --------------------------------------------------------------------------- #


def _seg(label: str, start: float, end: float, text: str, lang="en") -> TranscriptSegment:
    return TranscriptSegment(
        speaker_label=label, start=start, end=end, text=text,
        language=lang, confidence=0.93, source=AudioSourceKind.MICROPHONE,
    )


def _ready_report(ready: bool = True) -> SystemReadinessReport:
    return SystemReadinessReport(
        models=[], mic_permission=True, system_audio_permission=True,
        compute_backend="cpu", os_supports_process_tap=True, missing=[] if ready else ["x"],
    )


def _emit_sink() -> List[str]:
    """A list that doubles as the evaluate_js channel (collects JS strings)."""
    return []


_EVT = re.compile(r"window\.onBackendEvent\((.*)\)$", re.DOTALL)


def _events(sink: List[str]) -> List[dict]:
    """Parse the collected JS strings into event dicts (skip non-event JS)."""
    out: List[dict] = []
    for js in sink:
        m = _EVT.search(js.strip())
        if m:
            out.append(json.loads(m.group(1)))
    return out


def _make_fake_start(segments: List[TranscriptSegment], *, delay: float = 0.01):
    """A start_session fake: returns a live session that streams `segments` on a thread."""
    from meeting_asr.types import AudioSource
    from meeting_asr.session import TranscriptionSession

    def _start(*, sources=(AudioSourceKind.MICROPHONE,), language_hint=None,
               on_segment=None, on_error=None):
        session = TranscriptionSession(
            sources=[AudioSource(kind=k, enabled=True) for k in sources],
            language_hint=language_hint, on_segment=on_segment, on_error=on_error,
        )
        session.begin()  # ACTIVE

        def _stream():
            for s in segments:
                time.sleep(delay)
                session.deliver_segment(s)  # fans out to on_segment

        threading.Thread(target=_stream, name="fake-live", daemon=True).start()
        return session

    return _start


def _drain(sink, *, want_types=(), min_segments=0, timeout=2.0):
    """Wait until each `want_types` and `min_segments` segment events arrive (or timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        evs = _events(sink)
        type_ok = all(any(e["type"] == t for e in evs) for t in want_types)
        seg_ok = sum(1 for e in evs if e["type"] == "segment") >= min_segments
        if type_ok and seg_ok:
            return evs
        time.sleep(0.01)
    return _events(sink)


def _new_api(sink, *, ready=True, start_fn=None, **kw):
    from app.bridge import Api

    return Api(
        emit=sink.append,
        check_readiness_fn=lambda: _ready_report(ready),
        start_session_fn=start_fn or _make_fake_start([_seg("Speaker 1", 0.1, 0.9, "hello world")]),
        **kw,
    )


# --------------------------------------------------------------------------- #
# T017 — US1 contract: get_readiness shape
# --------------------------------------------------------------------------- #


def test_get_readiness_shape():
    api = _new_api(_emit_sink(), ready=True)
    dto = api.get_readiness()
    assert dto["ready"] is True
    for key in ("compute_backend", "os_supports_process_tap", "mic_permission",
                "system_audio_permission", "models", "missing"):
        assert key in dto
    assert dto["missing"] == []


# --------------------------------------------------------------------------- #
# T017 — US1 contract: start_live return shape + event payloads
# --------------------------------------------------------------------------- #


def test_start_live_returns_id_and_emits_segment_and_status():
    sink = _emit_sink()
    segs = [_seg("Speaker 1", 0.1, 0.9, "good morning", "en"),
            _seg("Speaker 2", 1.0, 1.8, "hello there", "en")]
    api = _new_api(sink, start_fn=_make_fake_start(segs))

    res = api.start_live(["microphone"])
    assert "app_session_id" in res and isinstance(res["app_session_id"], str)
    # Async contract: model loading runs on a worker, so start_live returns immediately
    # with `starting`; `starting` is emitted now and `recording` follows once capture begins.
    assert res.get("starting") is True

    # `starting` is emitted synchronously before the worker runs.
    evs = _drain(sink, want_types=("status",), timeout=2.0)
    assert any(e["type"] == "status" and e["status"] == "starting" for e in evs)
    # `recording` (and the streamed segments) arrive once the worker has begun capture.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        evs = _events(sink)
        if any(e["type"] == "status" and e["status"] == "recording" for e in evs) and \
           sum(1 for e in evs if e["type"] == "segment") >= 2:
            break
        time.sleep(0.01)
    status_ev = next((e for e in evs if e["type"] == "status" and e["status"] == "recording"), None)
    assert status_ev is not None, f"recording status not emitted; events={evs}"

    seg_evs = [e for e in evs if e["type"] == "segment"]
    assert len(seg_evs) >= 2
    # Segments arrive in finalized (chronological start) order.
    starts = [e["segment"]["start"] for e in seg_evs]
    assert starts == sorted(starts)
    # SegmentDTO carries every documented field.
    s0 = seg_evs[0]["segment"]
    for key in ("segment_id", "speaker_label", "start", "end", "text",
                "language", "confidence", "confidence_band", "source", "is_final"):
        assert key in s0
    assert s0["speaker_label"] == "Speaker 1"
    assert s0["source"] == "microphone"


def test_stop_session_is_idempotent_and_emits_stopped():
    sink = _emit_sink()
    api = _new_api(sink)
    api.start_live(["microphone"])
    _drain(sink, want_types=("segment",))

    r1 = api.stop_session()
    assert r1 == {"status": "stopped"}
    # Idempotent: a second stop does not raise and stays stopped.
    r2 = api.stop_session()
    assert r2["status"] == "stopped"
    # Status events are delivered asynchronously by the emit pump; wait for the
    # specific `stopped` status rather than just any status event.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        evs = _events(sink)
        if any(e["type"] == "status" and e["status"] == "stopped" for e in evs):
            break
        time.sleep(0.01)
    assert any(e["type"] == "status" and e["status"] == "stopped" for e in evs)


# --------------------------------------------------------------------------- #
# Live-emit decoupling: a slow webview evaluate_js must NOT block the producer
# (regression guard — segments were batching until Stop because the pipeline
# thread blocked on the main-thread-bound evaluate_js).
# --------------------------------------------------------------------------- #


def test_emit_does_not_block_producer_on_slow_sink():
    from app.bridge import Api

    release = threading.Event()
    delivered: List[str] = []

    def slow_emit(js: str) -> None:
        # Simulate evaluate_js blocking until the main thread is free.
        release.wait(timeout=5.0)
        delivered.append(js)

    api = Api(emit=slow_emit, check_readiness_fn=lambda: _ready_report(True),
              start_session_fn=_make_fake_start([_seg("Speaker 1", 0.1, 0.9, "hi")]))

    # Each _emit must return immediately even while slow_emit is parked.
    t0 = time.monotonic()
    for i in range(5):
        api._emit({"type": "status", "status": f"s{i}"})
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"_emit blocked on the slow sink ({elapsed:.2f}s)"
    assert delivered == []  # nothing delivered while the sink is parked

    # Once the sink is released, the pump drains everything in order.
    release.set()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and len(delivered) < 5:
        time.sleep(0.01)
    assert len(delivered) == 5
    statuses = [json.loads(_EVT.search(js.strip()).group(1))["status"] for js in delivered]
    assert statuses == ["s0", "s1", "s2", "s3", "s4"]  # FIFO order preserved


# --------------------------------------------------------------------------- #
# T017 — US1 contract: not_ready / session.busy rejections
# --------------------------------------------------------------------------- #


def test_start_live_rejects_not_ready():
    api = _new_api(_emit_sink(), ready=False)
    res = api.start_live(["microphone"])
    assert "error" in res
    assert res["error"]["code"] == "not_ready"


def test_start_live_rejects_when_busy():
    sink = _emit_sink()
    api = _new_api(sink)
    api.start_live(["microphone"])  # active
    res = api.start_live(["microphone"])
    assert "error" in res
    assert res["error"]["code"] == "session.busy"


# --------------------------------------------------------------------------- #
# T030 — US3 contract: get_readiness missing[] + prepare event sequence
# --------------------------------------------------------------------------- #


def test_get_readiness_surfaces_missing():
    api = _new_api(_emit_sink(), ready=False)
    dto = api.get_readiness()
    assert dto["ready"] is False
    assert dto["missing"] and len(dto["missing"]) >= 1


def test_prepare_emits_progress_then_done():
    from meeting_asr.types import ModelState, PrepareProgress

    ticks: list = []

    def _fake_prepare(*, progress=None):
        for asset in ("nemotron", "sortformer"):
            if progress:
                progress(PrepareProgress(asset=asset, downloaded=0, total=1, state=ModelState.DOWNLOADING))
                progress(PrepareProgress(asset=asset, downloaded=1, total=1, state=ModelState.CACHED))
        return _ready_report(True)

    sink = _emit_sink()
    api = _new_api(sink, prepare_models_fn=_fake_prepare)
    res = api.prepare()
    assert res == {"started": True}
    evs = _drain(sink, want_types=("prepare_progress", "prepare_done"))
    types = [e["type"] for e in evs]
    assert "prepare_progress" in types
    assert types.index("prepare_progress") < len(types)  # progress precedes done
    assert any(e["type"] == "prepare_done" and e["readiness"]["ready"] is True for e in evs)


# --------------------------------------------------------------------------- #
# T017 — get_transcript snapshot
# --------------------------------------------------------------------------- #


def test_get_transcript_snapshot_after_stop():
    sink = _emit_sink()
    segs = [_seg("Speaker 1", 0.1, 0.9, "one", "en")]
    api = _new_api(sink, start_fn=_make_fake_start(segs))
    api.start_live(["microphone"])
    _drain(sink, want_types=("segment",))
    api.stop_session()
    snap = api.get_transcript()
    assert len(snap["segments"]) == 1
    assert snap["segments"][0]["text"] == "one"
    assert any(s["label"] == "Speaker 1" for s in snap["speakers"])


# --------------------------------------------------------------------------- #
# T040 — US4 export bridge wiring
# --------------------------------------------------------------------------- #


def test_export_transcript_writes_and_returns_path(tmp_path):
    from pathlib import Path

    written = {}

    def fake_export(path, segments, speakers, *, session_meta=None):
        Path(path).write_text("ok", encoding="utf-8")
        written["path"] = path
        written["n"] = len(segments)
        return path

    sink = _emit_sink()
    segs = [_seg("Speaker 1", 0.1, 0.9, "export me", "en")]
    out = tmp_path / "t.md"
    api = _new_api(
        sink,
        start_fn=_make_fake_start(segs),
        export_fn=fake_export,
        pick_save_path=lambda fmt: str(out),
    )
    api.start_live(["microphone"])
    _drain(sink, want_types=("segment",))
    api.stop_session()

    res = api.export_transcript(str(out), "markdown")
    assert res == {"path": str(out)}
    assert written["n"] == 1 and written["path"] == str(out)


def test_export_empty_returns_error():
    api = _new_api(_emit_sink())  # no session produced yet
    res = api.export_transcript("/out.md", "markdown")
    assert "error" in res and res["error"]["code"] == "export.empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
