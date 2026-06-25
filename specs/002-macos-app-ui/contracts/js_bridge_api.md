# Contract — JS ↔ Python WebView Bridge

Module: `app/bridge.py` (the `Api` class mounted as `pywebview` `js_api`) and the
`app/web/app.js` event handler. This is the **only** interface between the UI and the
backend (Constitution VII). All payloads are JSON-serializable plain objects.

## Calling convention

- **JS → Python (requests)**: `await window.pywebview.api.<method>(args...)`. Each method
  returns a JSON-serializable result (or `{ "error": ErrorInfo }`). Long-running methods
  return immediately (an accepted/started ack) and report via events.
- **Python → JS (events)**: the bridge calls
  `window.evaluate_js("window.onBackendEvent(" + json + ")")` from worker threads. The UI
  implements a single `window.onBackendEvent(evt)` dispatcher keyed by `evt.type`.
- **Threading**: no request method blocks the WebView UI thread on inference. `prepare`,
  `start_live`, and `transcribe_file` start work on a Python worker thread.

## Request methods (JS → Python)

| Method | Args | Returns | Notes |
|--------|------|---------|-------|
| `get_readiness()` | — | `ReadinessDTO` | Wraps `check_readiness()`; never raises (FR-009/013) |
| `prepare()` | — | `{ "started": true }` | Runs `prepare_models(progress=…)` on a worker; emits `prepare_progress` + `prepare_done` events (FR-008/010) |
| `start_live(sources, language_hint)` | `sources: ["microphone","system"]`, `language_hint: str\|null` | `{ "app_session_id": str }` or `{ "error": ErrorInfo }` | Wraps `start_session`; emits `segment`/`error`/`status` events (US1) |
| `stop_session()` | — | `{ "status": "stopped" }` | Wraps `session.stop()`; idempotent (US1 sc.5) |
| `transcribe_file(path, language_hint)` | `path: str`, `language_hint: str\|null` | `{ "app_session_id": str }` or `{ "error": ErrorInfo }` | Wraps `transcribe_file`; emits `segment`/`progress`/`status`/`error` (US2) |
| `pick_audio_file()` | — | `{ "path": str\|null }` | Native open dialog (audio types); `null` if cancelled |
| `pick_export_path(format)` | `format: "markdown"\|"json"` | `{ "path": str\|null }` | Native save dialog with the right extension |
| `export_transcript(path, format)` | `path`, `format` | `{ "path": str }` or `{ "error": ErrorInfo }` | Writes via `meeting_asr.export` (US4, FR-013) |
| `get_transcript()` | — | `{ "segments": SegmentDTO[], "speakers": SpeakerDTO[] }` | Snapshot of current session (for re-render) |

## Events (Python → JS, via `onBackendEvent`)

Each event is `{ "type": <name>, ...payload }`:

| `type` | Payload | Trigger |
|--------|---------|---------|
| `prepare_progress` | `{ asset, downloaded, total, fraction, state }` | per `PrepareProgress` tick |
| `prepare_done` | `{ readiness: ReadinessDTO }` | download finished (or failed → `error`) |
| `segment` | `{ segment: SegmentDTO }` | per finalized `TranscriptSegment` (`on_segment`) — drives live render (SC-003) |
| `progress` | `{ fraction: float }` | file-import progress (US2) |
| `status` | `{ status: AppStatus }` | AppSession status change (setting_up/ready/recording/processing/stopping/stopped/error) |
| `error` | `{ error: ErrorInfo }` | recoverable (`session.notify`) or terminal (`set_error`) condition (FR-014) |

## DTO shapes

```
ReadinessDTO = {
  ready: bool, compute_backend: str, os_supports_process_tap: bool,
  mic_permission: bool, system_audio_permission: bool,
  models: [{ name, kind, state, is_cached }],
  missing: [str]              // plain-language items rendered on the setup screen
}
SegmentDTO = {
  segment_id, speaker_label, start, end, text,
  language: str|null, confidence: float, confidence_band: str,
  source: "microphone"|"system"|null, is_final: bool
}
SpeakerDTO = { label, color, total_speech_seconds, segment_count }
ErrorInfo  = { code, message, recoverable: bool, hint: str|null }
AppStatus  = "setting_up"|"ready"|"recording"|"processing"|"stopping"|"stopped"|"error"
```

## Behavioral guarantees

- `get_readiness` / `prepare_done` MUST surface the exact `missing[]` list so the setup
  screen can explain each unmet item (FR-008/009).
- `start_live` and `transcribe_file` MUST reject with `ErrorInfo(code="not_ready")` when
  `check_readiness().ready` is false, and with `code="session.busy"` if one is active
  (FR-020) — the UI never starts a session over an active one.
- `segment` events MUST arrive in finalized order and preserve chronological `start`
  (FR-019); the UI appends/renders without reordering beyond `start`.
- No method or event may carry audio samples or transmit data off-device (Principle I).
- A consumer-side exception in JS event handling MUST NOT affect the backend (the bridge
  fire-and-forgets `evaluate_js`).

## Testability (offline)

`tests/contract/test_bridge_api.py` instantiates `Api` with injected backend fakes
(reusing `tests/_fakes`) and a stub `evaluate_js` sink, then asserts: method
return-shapes, event sequence/ordering for a fake live run and a fake file run, and the
not-ready / busy rejections — all without opening a WKWebView window.
