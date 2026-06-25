"""Logging + error infrastructure (task T011).

Centralizes:
  * the exception hierarchy surfaced by the public API and capture backends, and
  * ``ErrorInfo`` factory helpers that pair each failure mode with an actionable
    hint (FR-015, FR-018, FR-021), and
  * a tiny structured logger (stdlib ``logging`` + JSON-ish records) so a UI/test
    can consume machine-readable diagnostics.

Kept dependency-free (stdlib + ``types``) so it is importable everywhere.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Optional

from .types import AudioSourceKind, CaptureState, ErrorInfo, ModelState

# --------------------------------------------------------------------------- #
# Exception hierarchy
# --------------------------------------------------------------------------- #


class MeetingAsrError(Exception):
    """Base for all meeting_asr errors."""


class CaptureError(MeetingAsrError):
    """Base for audio-capture failures (carries the source kind)."""

    def __init__(self, message: str, *, source: Optional[AudioSourceKind] = None) -> None:
        super().__init__(message)
        self.source = source


class CapturePermissionError(CaptureError):
    """A required capture permission is missing (FR-015)."""


class CaptureDeviceError(CaptureError):
    """A capture device was lost or is unavailable mid-session."""


class ReadinessError(MeetingAsrError):
    """Raised by ``start_session`` when required models/permissions are missing."""


class SessionBusyError(MeetingAsrError):
    """A new session was started while another is ACTIVE (FR-020)."""


class ModelError(MeetingAsrError):
    """Model download/cache failed (interrupted, corrupt, checksum mismatch)."""


# --------------------------------------------------------------------------- #
# ErrorInfo factories (each pairs a failure with an actionable hint)
# --------------------------------------------------------------------------- #


def permission_error_info(source: AudioSourceKind, *, detail: str = "") -> ErrorInfo:
    """Actionable permission error for mic vs. system audio (FR-015)."""
    if source is AudioSourceKind.MICROPHONE:
        hint = (
            "Grant microphone access: System Settings → Privacy & Security → "
            "Microphone → enable your terminal/app, then restart the session."
        )
    else:
        hint = (
            "Grant system-audio capture: approve the capture prompt on first run "
            "(macOS 14.4+). System Settings → Privacy & Security → Screen & System "
            "Audio Recording → enable your terminal/app."
        )
    msg = f"{source.value} capture permission denied"
    if detail:
        msg = f"{msg}: {detail}"
    return ErrorInfo(code="CAPTURE_PERMISSION_DENIED", message=msg, recoverable=True, hint=hint)


def device_error_info(source: AudioSourceKind, *, detail: str = "") -> ErrorInfo:
    """Device lost / unavailable mid-session (edge case)."""
    msg = f"{source.value} capture device unavailable"
    if detail:
        msg = f"{msg}: {detail}"
    return ErrorInfo(
        code="CAPTURE_DEVICE_UNAVAILABLE",
        message=msg,
        recoverable=True,
        hint="Check that the device is connected and not in use by another app, then retry.",
    )


def lag_error_info(detail: str, *, dropped: bool = False) -> ErrorInfo:
    """Compute/memory pressure → graceful degradation with lag signaling (FR-021).

    ``dropped`` should normally be False: the pipeline buffers and signals lag
    rather than dropping audio. If a frame ever *is* dropped, the hint escalates.
    """
    return ErrorInfo(
        code="COMPUTE_PRESSURE",
        message=f"real-time inference cannot keep pace: {detail}",
        recoverable=True,
        hint=(
            "Transcription is lagging behind realtime; audio is buffered (not dropped). "
            "Reduce concurrent load or move to a higher-throughput backend."
        )
        if not dropped
        else ("Audio frames were dropped to avoid timeline corruption; reduce load or "
              "use a faster precision/backend."),
    )


def model_error_info(asset: str, *, detail: str, recoverable: bool = True) -> ErrorInfo:
    """Model download/cache failure (interrupted download edge case)."""
    return ErrorInfo(
        code="MODEL_ERROR",
        message=f"model '{asset}' unavailable: {detail}",
        recoverable=recoverable,
        hint="Re-run prepare_models(); downloads are resumable and no corrupt cache is left.",
    )


def unsupported_language_info(detail: str = "") -> ErrorInfo:
    """Unsupported/unexpected language → best-effort + low confidence (FR-018)."""
    return ErrorInfo(
        code="UNSUPPORTED_LANGUAGE",
        message=f"speech in an unsupported or unrecognizable language: {detail}".rstrip(": "),
        recoverable=True,
        hint="The segment is transcribed best-effort and flagged LOW/UNKNOWN confidence.",
    )


# --------------------------------------------------------------------------- #
# Structured logger
# --------------------------------------------------------------------------- #

_CONFIGURED = False


class _StructuredFormatter(logging.Formatter):
    """Emits one JSON object per record (best-effort; non-JSON fields stringified)."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach extra structured fields if the caller passed them.
        for key, value in record.__dict__.items():
            if key in payload or key.startswith("_") or key in {
                "args", "msg", "name", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "taskName",
            }:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO, *, force: bool = False) -> None:
    """Install the structured formatter on the root handler (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    root = logging.getLogger("meeting_asr")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_StructuredFormatter())
        root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "meeting_asr") -> logging.Logger:
    """Return a child logger under the ``meeting_asr`` namespace (auto-configured)."""
    if not name.startswith("meeting_asr"):
        name = f"meeting_asr.{name}"
    configure_logging()
    return logging.getLogger(name)


def log_error_info(logger: logging.Logger, info: ErrorInfo, *, context: Optional[str] = None) -> None:
    """Emit an ``ErrorInfo`` as a structured WARN/ERROR record."""
    extra = {"code": info.code, "recoverable": info.recoverable, "hint": info.hint}
    if context:
        extra["context"] = context
    logger.error("%s", info.message, extra=extra)


__all__ = [
    "MeetingAsrError",
    "CaptureError",
    "CapturePermissionError",
    "CaptureDeviceError",
    "ReadinessError",
    "SessionBusyError",
    "ModelError",
    "permission_error_info",
    "device_error_info",
    "lag_error_info",
    "model_error_info",
    "unsupported_language_info",
    "configure_logging",
    "get_logger",
    "log_error_info",
]
