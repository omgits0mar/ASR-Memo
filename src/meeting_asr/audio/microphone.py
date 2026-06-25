"""MicrophoneCapture — local mic via PortAudio/sounddevice (task T019).

Opens the default input device, captures in its native format, and resamples to
16 kHz mono float32 via :class:`AudioMixer`, emitting canonical ``AudioFrame``s
tagged ``source=MICROPHONE`` with session-clock timestamps.

``sounddevice`` is lazy-imported so the module loads (and the protocol is
satisfiable) without PortAudio installed; the capture only runs on real hardware.
Permission/device failures map to actionable exceptions (FR-015).
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

from .._logging import CaptureDeviceError, CapturePermissionError, get_logger
from ..types import AudioFrame, AudioSourceKind, CaptureState
from .mixer import SAMPLE_RATE, AudioMixer

_log = get_logger("audio.microphone")


class MicrophoneCapture:
    """AudioCapture-conforming microphone backend (PortAudio)."""

    kind = AudioSourceKind.MICROPHONE

    def __init__(self, *, device: Optional[int] = None, block_seconds: float = 0.1) -> None:
        self._device = device
        self._block_seconds = block_seconds
        self._state = CaptureState.IDLE
        self._stream = None
        self._mixer = AudioMixer()
        self._on_frame: Optional[Callable[[AudioFrame], None]] = None
        self._lock = threading.Lock()

    # ---- AudioCapture protocol ----

    def permission_status(self) -> bool:
        """Best-effort: is a default input device present?

        macOS TCC mic permission cannot be queried without prompting; a denial
        surfaces as ``CapturePermissionError`` on :meth:`start`.
        """
        try:
            import sounddevice as sd

            return sd.default.device[0] is not None  # [0] = input device index
        except Exception:
            return False

    def state(self) -> CaptureState:
        return self._state

    def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        try:
            import sounddevice as sd
        except ImportError as e:  # PortAudio/sounddevice not installed
            raise CaptureDeviceError(
                "sounddevice/PortAudio not installed (run: brew install portaudio && pip install sounddevice)",
                source=self.kind,
            ) from e

        self._on_frame = on_frame
        try:
            info = sd.query_devices(kind="input")
        except Exception as e:
            raise CaptureDeviceError(f"no input device available: {e}", source=self.kind) from e

        native_rate = int(info["default_samplerate"])
        channels = int(info["max_input_channels"]) or 1
        blocksize = int(self._block_seconds * native_rate)

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
            if status and getattr(status, "input_overflow", False):
                _log.warning("mic input overflow (frame dropped by PortAudio)")
            try:
                frame = self._mixer.feed(indata.copy(), native_rate, source=self.kind, channels=channels)
                if self._on_frame is not None:
                    self._on_frame(frame)
            except Exception:
                _log.exception("mic capture frame error")

        try:
            self._stream = sd.InputStream(
                device=self._device,
                samplerate=native_rate,
                channels=channels,
                dtype="float32",
                blocksize=blocksize,
                callback=_callback,
            )
            self._stream.start()
            self._state = CaptureState.CAPTURING
            _log.info("microphone capturing @ %d Hz, %d ch → 16 kHz mono", native_rate, channels)
        except Exception as e:
            msg = str(e).lower()
            if "permission" in msg or "denied" in msg or "not authorized" in msg:
                raise CapturePermissionError(str(e), source=self.kind) from e
            # PortAudio often reports permission denial opaquely; surface a hint.
            if "could not" in msg or "invalid" in msg:
                raise CapturePermissionError(
                    f"microphone capture failed (check macOS mic permission): {e}", source=self.kind
                ) from e
            raise CaptureDeviceError(f"microphone start failed: {e}", source=self.kind) from e

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                _log.exception("error closing mic stream")
        self._state = CaptureState.IDLE


__all__ = ["MicrophoneCapture"]
