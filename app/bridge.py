"""Api — the JS↔Python webview bridge (task T010 + US1–US4 methods).

Mounted as ``pywebview``'s ``js_api``: JS calls ``await window.pywebview.api.<m>(...)``;
the backend pushes events back via ``window.evaluate_js("window.onBackendEvent(...)")``
from worker threads (research Decision 2). No method blocks the UI thread on
inference — ``prepare``, ``transcribe_file``, and the long tail run on workers;
``start_live`` returns once capture has begun and streams segments as events.

Every backend dependency is an injectable seam (``*_fn`` / pickers) so the whole
surface is contract-tested **headlessly** with fakes and a stub ``evaluate_js`` —
no WKWebView window, no models, no network (Constitution I + VII).
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Callable, List, Optional

from meeting_asr._logging import CapturePermissionError, ReadinessError, SessionBusyError
from meeting_asr.types import AudioSourceKind, ErrorInfo

from .app_session import AppSession, AppStatus, InputMode
from .dto import error_dto, prepare_progress_dto, readiness_dto, segment_dto, speaker_dto

__all__ = ["Api"]

_BUSY = ErrorInfo(
    code="session.busy",
    message="a session is already running; stop it before starting another",
    recoverable=False,
    hint="Click Stop, then start a new session.",
)
_NOT_READY = ErrorInfo(
    code="not_ready",
    message="the backend is not ready (models or permissions missing)",
    recoverable=False,
    hint="Complete first-run setup: download models and grant microphone permission.",
)


def _to_source_kind(s) -> AudioSourceKind:
    if isinstance(s, AudioSourceKind):
        return s
    low = str(s).strip().lower()
    if low in ("microphone", "mic"):
        return AudioSourceKind.MICROPHONE
    if low in ("system", "system_audio", "systemaudio"):
        return AudioSourceKind.SYSTEM
    raise ValueError(f"unknown audio source: {s!r}")


def _ext_format(path: str) -> str:
    low = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if low in ("md", "markdown"):
        return "markdown"
    if low == "json":
        return "json"
    raise ValueError(f"unknown export extension for {path!r} (use .md or .json)")


class Api:
    """The single bridge object exposed to the webview as ``window.pywebview.api``."""

    def __init__(
        self,
        *,
        emit: Optional[Callable[[str], None]] = None,
        start_session_fn: Optional[Callable] = None,
        check_readiness_fn: Optional[Callable[[], object]] = None,
        prepare_models_fn: Optional[Callable] = None,
        transcribe_file_fn: Optional[Callable] = None,
        export_fn: Optional[Callable] = None,
        pick_open_path: Optional[Callable[[], Optional[str]]] = None,
        pick_save_path: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self._emit_raw = emit
        self._start_session_fn = start_session_fn
        self._check_readiness_fn = check_readiness_fn
        self._prepare_models_fn = prepare_models_fn
        self._transcribe_file_fn = transcribe_file_fn
        self._export_fn = export_fn
        self._pick_open_path = pick_open_path
        self._pick_save_path = pick_save_path
        self._window = None
        self._app_session: Optional[AppSession] = None
        self._workers: List[threading.Thread] = []
        # Emit pump: live segments are produced on the pipeline's background
        # threads, but the webview's evaluate_js blocks the caller until the macOS
        # main thread runs the script. Pushing directly from the pipeline thread
        # serializes (and stalls) live updates so they only flush at Stop. Decouple
        # via a queue drained by one daemon thread — producers enqueue and return
        # instantly; the pump owns the (possibly blocking) evaluate_js call.
        self._emit_queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._emit_pump: Optional[threading.Thread] = None
        self._emit_pump_lock = threading.Lock()

    # ---- wiring ----

    def attach(self, window) -> None:
        """Bind the real webview window: emit → ``window.evaluate_js`` + native pickers."""
        self._window = window
        if self._emit_raw is None:
            self._emit_raw = lambda js: window.evaluate_js(js)

    # ---- lazy default resolvers (production wiring) ----

    def _start(self, **kw):
        fn = self._start_session_fn
        if fn is None:
            import meeting_asr
            fn = meeting_asr.start_session
        return fn(**kw)

    def _readiness(self):
        fn = self._check_readiness_fn
        if fn is None:
            import meeting_asr
            fn = meeting_asr.check_readiness
        return fn()

    def _prepare(self, **kw):
        fn = self._prepare_models_fn
        if fn is None:
            import meeting_asr
            fn = meeting_asr.prepare_models
        return fn(**kw)

    def _transcribe(self, **kw):
        fn = self._transcribe_file_fn
        if fn is None:
            import meeting_asr
            fn = meeting_asr.transcribe_file
        return fn(**kw)

    def _export(self, path, segments, speakers, *, session_meta=None):
        fn = self._export_fn
        if fn is None:
            from meeting_asr.export import write_export
            fn = write_export
        return fn(path, segments, speakers, session_meta=session_meta)

    # ---- event channel ----

    def _emit(self, event: dict) -> None:
        js = "window.onBackendEvent(" + json.dumps(event, ensure_ascii=False) + ")"
        if self._emit_raw is None:
            return  # no window / no sink yet — headless without a channel
        # Enqueue and return immediately so a producer (pipeline thread) is never
        # blocked on the main-thread-bound evaluate_js. The pump preserves order.
        self._ensure_emit_pump()
        self._emit_queue.put(js)

    def _ensure_emit_pump(self) -> None:
        """Start the single emit-pump daemon on first use (idempotent)."""
        with self._emit_pump_lock:
            if self._emit_pump is not None and self._emit_pump.is_alive():
                return
            t = threading.Thread(target=self._emit_loop, name="api-emit-pump", daemon=True)
            self._emit_pump = t
            t.start()

    def _emit_loop(self) -> None:
        """Drain queued JS strings to the webview, one at a time, in order."""
        while True:
            js = self._emit_queue.get()
            if js is None:  # sentinel (unused today; reserved for clean shutdown)
                return
            emit = self._emit_raw
            if emit is None:
                continue
            try:
                emit(js)
            except Exception:  # a consumer-side error must not kill the pump
                pass

    def _start_worker(self, target: Callable) -> None:
        t = threading.Thread(target=target, name="api-worker", daemon=True)
        self._workers.append(t)
        t.start()

    # ================================================================ #
    # Request methods (JS → Python)
    # ================================================================ #

    # ---- US3: setup / readiness ----

    def get_readiness(self) -> dict:
        """Wrap check_readiness(); never raises (FR-009/013)."""
        try:
            report = self._readiness()
        except Exception as e:  # defensive: the UI must always get an answer
            return {
                "ready": False, "compute_backend": "cpu",
                "os_supports_system_audio": False, "os_supports_process_tap": False,
                "mic_permission": False, "system_audio_permission": False,
                "models": [], "missing": [f"readiness check failed: {e}"],
            }
        return readiness_dto(report)

    def prepare(self) -> dict:
        """Run prepare_models on a worker; emit prepare_progress + prepare_done."""
        def _work() -> None:
            try:
                report = self._prepare(progress=self._on_prepare_progress)
                self._emit({"type": "prepare_done", "readiness": readiness_dto(report)})
            except Exception as e:
                self._emit({
                    "type": "error",
                    "error": error_dto(ErrorInfo(
                        code="prepare.failed", message=f"model setup failed: {e}",
                        recoverable=True, hint="Re-run Download models; downloads are resumable.",
                    )),
                })

        self._start_worker(_work)
        return {"started": True}

    def _on_prepare_progress(self, p) -> None:
        self._emit({"type": "prepare_progress", **prepare_progress_dto(p)})

    # ---- US1: live session ----

    def start_live(self, sources, language_hint=None) -> dict:
        """Begin a live diarized session; emit segment/status/error (US1).

        Model loading (the diarizer's CoreML compile can take ~1 min cold) runs on a
        worker so the UI stays responsive — this returns immediately with a
        ``starting`` status; ``recording`` (or ``error``) is emitted once capture
        has actually begun.
        """
        if self._app_session is not None and self._app_session.is_active():
            return {"error": error_dto(_BUSY)}
        try:
            report = self._readiness()
        except Exception:
            report = None
        if report is None or not getattr(report, "ready", False):
            return {"error": error_dto(_NOT_READY)}

        try:
            kinds = [_to_source_kind(s) for s in (sources or (AudioSourceKind.MICROPHONE,))]
        except ValueError as e:
            return {"error": error_dto(ErrorInfo(
                code="sources.invalid", message=str(e), recoverable=True, hint="Choose Microphone and/or System audio."))}

        app = AppSession(input_mode=InputMode.LIVE, source_kinds=tuple(kinds), language_hint=language_hint)
        app.starting()
        self._app_session = app  # claim before the worker (busy guard)
        on_segment = self._make_on_segment(app)
        on_error = self._make_on_error(app)
        self._emit({"type": "status", "status": AppStatus.STARTING.value})

        def _work() -> None:
            try:
                backend = self._start(sources=kinds, language_hint=language_hint,
                                      on_segment=on_segment, on_error=on_error)
            except SessionBusyError:
                self._fail_session(app, _BUSY)
                return
            except ReadinessError as e:
                self._fail_session(app, ErrorInfo(code="not_ready", message=str(e), recoverable=False, hint=_NOT_READY.hint))
                return
            except CapturePermissionError as e:
                self._fail_session(app, ErrorInfo(
                    code="capture.permission", message=str(e), recoverable=True,
                    hint="Grant microphone access in System Settings → Privacy → Microphone, then retry."))
                return
            except Exception as e:  # pragma: no cover - model/capture dependent
                self._fail_session(app, ErrorInfo(
                    code="start.failed", message=f"could not start session: {e}", recoverable=True,
                    hint="Retry; if it persists, re-run Download models or relaunch the app."))
                return
            # User may have clicked Stop during the load window — tear down cleanly.
            if app.status in (AppStatus.STOPPED, AppStatus.ERROR, AppStatus.STOPPING):
                try:
                    backend.stop()
                except Exception:
                    pass
                return
            app.begin(backend)
            self._emit({"type": "status", "status": AppStatus.RECORDING.value})

        self._start_worker(_work)
        return {"app_session_id": app.app_session_id, "starting": True}

    def stop_session(self) -> dict:
        """Halt the active session; idempotent; emit status=stopped (US1)."""
        app = self._app_session
        if app is None:
            return {"status": AppStatus.STOPPED.value}
        if app.backend_session is not None and app.is_active():
            app.stopping()
            self._emit({"type": "status", "status": AppStatus.STOPPING.value})
            try:
                app.backend_session.stop()
            except Exception as e:  # never let a stop failure strand the UI
                self._emit({"type": "error", "error": error_dto(ErrorInfo(
                    code="stop.failed", message=f"error stopping session: {e}",
                    recoverable=True, hint="The session transcript is retained for review."))})
        app.stopped()
        self._emit({"type": "status", "status": AppStatus.STOPPED.value})
        return {"status": AppStatus.STOPPED.value}

    # ---- US2: file import ----

    def transcribe_file(self, path, language_hint=None) -> dict:
        """Run the file-import pipeline on a worker; emit segment/progress/status/error."""
        if self._app_session is not None and self._app_session.is_active():
            return {"error": error_dto(_BUSY)}
        try:
            report = self._readiness()
        except Exception:
            report = None
        if report is None or not getattr(report, "ready", False):
            return {"error": error_dto(_NOT_READY)}

        app = AppSession(input_mode=InputMode.FILE, file_path=path, language_hint=language_hint)
        self._app_session = app  # claimed before the worker starts (busy guard)
        app.claim()  # actively claimed → is_active() True through the worker-startup window
        on_segment = self._make_on_segment(app)
        on_error = self._make_on_error(app)

        def _on_progress(frac: float) -> None:
            app.set_progress(frac)
            self._emit({"type": "progress", "fraction": float(frac)})

        def _work() -> None:
            self._emit({"type": "status", "status": AppStatus.PROCESSING.value})
            try:
                backend = self._transcribe(path=path, language_hint=language_hint,
                                           on_segment=on_segment, on_error=on_error,
                                           on_progress=_on_progress)
                app.backend_session = backend
                if getattr(backend, "error", None) is not None:
                    app.set_error(backend.error)
                    self._emit({"type": "error", "error": error_dto(backend.error)})
                    self._emit({"type": "status", "status": AppStatus.ERROR.value})
                else:
                    app.stopped()
                    self._emit({"type": "status", "status": AppStatus.STOPPED.value})
            except ReadinessError as e:
                self._fail_session(app, ErrorInfo(code="not_ready", message=str(e), recoverable=False, hint=_NOT_READY.hint))
            except Exception as e:
                self._fail_session(app, ErrorInfo(
                    code="audio.unreadable" if "audio" in str(e).lower() else "file.failed",
                    message=str(e), recoverable=True, hint="Try a WAV/FLAC/MP3 speech file."))

        self._start_worker(_work)
        return {"app_session_id": app.app_session_id}

    # ---- US4: export ----

    def pick_audio_file(self) -> dict:
        return {"path": self._pick_open()}

    def pick_export_path(self, format) -> dict:
        return {"path": self._pick_save(format)}

    def export_transcript(self, path, format=None) -> dict:
        app = self._app_session
        if app is None or not app.segments:
            return {"error": error_dto(ErrorInfo(
                code="export.empty", message="no transcript to export",
                recoverable=True, hint="Produce a transcript first, then export."))}
        try:
            _ = format or _ext_format(path)
        except ValueError as e:
            return {"error": error_dto(ErrorInfo(code="export.format", message=str(e), recoverable=True, hint="Choose Markdown (.md) or JSON (.json)."))}
        try:
            self._export(path, app.transcript_snapshot(), dict(app.speakers), session_meta=app.session_meta())
        except ValueError as e:
            return {"error": error_dto(ErrorInfo(code="export.format", message=str(e), recoverable=True, hint="Choose Markdown (.md) or JSON (.json)."))}
        except PermissionError as e:
            return {"error": error_dto(ErrorInfo(code="export.permission", message=str(e), recoverable=True, hint="Choose a location you can write to (check file/folder permissions)."))}
        except OSError as e:
            return {"error": error_dto(ErrorInfo(code="export.write_failed", message=str(e), recoverable=True, hint="Could not write the file — check disk space and the chosen path."))}
        except Exception as e:
            return {"error": error_dto(ErrorInfo(code="export.failed", message=str(e), recoverable=True, hint="Export failed; try a different location or format."))}
        return {"path": path}

    # ---- shared snapshot ----

    def get_transcript(self) -> dict:
        app = self._app_session
        if app is None:
            return {"segments": [], "speakers": []}
        return {
            "segments": [segment_dto(s) for s in app.transcript_snapshot()],
            "speakers": [speaker_dto(s, color=c, segment_count=n) for (s, c, n) in app.speakers_view()],
        }

    # ================================================================ #
    # Shared callbacks + helpers
    # ================================================================ #

    def _make_on_segment(self, app: AppSession):
        def _on_segment(seg) -> None:
            app.add_segment(seg)
            self._emit({"type": "segment", "segment": segment_dto(seg)})
        return _on_segment

    def _make_on_error(self, app: AppSession):
        def _on_error(info: ErrorInfo) -> None:
            if not info.recoverable:
                app.set_error(info)
                self._emit({"type": "status", "status": AppStatus.ERROR.value})
            self._emit({"type": "error", "error": error_dto(info)})
        return _on_error

    def _fail_session(self, app: AppSession, info: ErrorInfo) -> None:
        app.set_error(info)
        self._emit({"type": "error", "error": error_dto(info)})
        self._emit({"type": "status", "status": AppStatus.ERROR.value})

    # ---- native pickers (default: the webview file dialogs) ----

    def _pick_open(self) -> Optional[str]:
        if self._pick_open_path is not None:
            return self._pick_open_path()
        return self._dialog_open()

    def _pick_save(self, format: str) -> Optional[str]:
        if self._pick_save_path is not None:
            return self._pick_save_path(format)
        return self._dialog_save(format)

    def _dialog_open(self) -> Optional[str]:  # pragma: no cover - needs a window
        if self._window is None:
            return None
        result = self._window.create_file_dialog(
            webview_dialog_type="open_file", file_types=("Audio Files (*.wav;*.flac;*.mp3;*.m4a;*.ogg)",),
        )
        return _first_dialog_path(result)

    def _dialog_save(self, format: str) -> Optional[str]:  # pragma: no cover - needs a window
        if self._window is None:
            return None
        ext = ".md" if format == "markdown" else ".json"
        result = self._window.create_file_dialog(webview_dialog_type="save_file", save_filename=f"transcript{ext}")
        return _first_dialog_path(result)


def _first_dialog_path(result) -> Optional[str]:
    """Normalize a pywebview dialog result (str | list | tuple | None) to a path or None."""
    if result is None:
        return None
    if isinstance(result, (list, tuple)):
        return result[0] if result else None
    return str(result)
