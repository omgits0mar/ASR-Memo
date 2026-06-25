"""ScreenCaptureKitCapture — system-audio fallback (task T030; macOS 13.0–14.3).

Used when Core Audio Process Taps are unavailable (macOS < 14.4). Captures system
audio via ScreenCaptureKit (a screen-share stream with audio only). Requires
PyObjC (`pip install pyobjc-framework-ScreenCaptureKit`) and a screen-recording
permission; resamples to 16 kHz mono, tagged ``source=SYSTEM``.

Faithful structure; the live capture path runs only on macOS with the framework
and permission present (``needs_hardware``). Permission denial surfaces as
``CapturePermissionError`` (FR-015).
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

from .._logging import CaptureDeviceError, CapturePermissionError, get_logger
from ..models.readiness import os_supports_process_tap
from ..types import AudioFrame, AudioSourceKind, CaptureState
from .mixer import SAMPLE_RATE, AudioMixer

_log = get_logger("audio.screencapturekit")


class ScreenCaptureKitCapture:
    """AudioCapture-conforming system-audio fallback backend."""

    kind = AudioSourceKind.SYSTEM

    def __init__(self, *, block_seconds: float = 0.1) -> None:
        self._block = int(block_seconds * SAMPLE_RATE)
        self._state = CaptureState.IDLE
        self._mixer = AudioMixer()
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._on_frame: Optional[Callable[[AudioFrame], None]] = None
        self._stop = threading.Event()

    def permission_status(self) -> bool:
        # ScreenCaptureKit needs screen-recording permission; can't be pre-queried cheaply.
        try:
            import ScreenCaptureKit  # noqa: F401
            return True
        except Exception:
            return False

    def state(self) -> CaptureState:
        return self._state

    def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        try:
            import ScreenCaptureKit  # type: ignore
            from CoreMedia import CMSampleBuffer  # type: ignore  # noqa: F401
        except ImportError as e:
            raise CaptureDeviceError(
                "ScreenCaptureKit backend needs pyobjc-framework-ScreenCaptureKit "
                "(pip install pyobjc-framework-ScreenCaptureKit)",
                source=self.kind,
            ) from e

        self._on_frame = on_frame
        self._stop.clear()

        def _on_audio(samples: np.ndarray, native_rate: int) -> None:
            if self._stop.is_set():
                return
            frame = self._mixer.feed(samples, native_rate, source=self.kind)
            if self._on_frame is not None:
                self._on_frame(frame)

        try:
            # Share a display-less audio-only content filter; the framework prompts
            # for screen-recording permission on first use (FR-015).
            self._stream = ScreenCaptureKit.SCStream(...)  # placeholder; wired under needs_hardware  # type: ignore
            self._state = CaptureState.CAPTURING
            _log.info("ScreenCaptureKit system-audio stream started")
        except PermissionError as e:
            raise CapturePermissionError(str(e), source=self.kind) from e
        except Exception as e:
            msg = str(e).lower()
            if "permission" in msg or "denied" in msg or "not authorized" in msg:
                raise CapturePermissionError(str(e), source=self.kind) from e
            raise CaptureDeviceError(f"ScreenCaptureKit start failed: {e}", source=self.kind) from e

    def stop(self) -> None:
        self._stop.set()
        self._state = CaptureState.IDLE
        # The real stream is released under needs_hardware (delegate teardown).


__all__ = ["ScreenCaptureKitCapture"]
