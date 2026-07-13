"""Windows system-audio capture via WASAPI loopback."""

from __future__ import annotations

import platform
import threading
from typing import Callable, Optional

import numpy as np

from .._logging import CaptureDeviceError, CapturePermissionError, get_logger
from ..types import AudioFrame, AudioSourceKind, CaptureState
from .mixer import AudioMixer

_log = get_logger("audio.wasapi_loopback")


class WasapiLoopbackCapture:
    """AudioCapture-conforming Windows loopback backend."""

    kind = AudioSourceKind.SYSTEM

    def __init__(self, *, device: Optional[int] = None, block_seconds: float = 0.1) -> None:
        self._device = device
        self._block_seconds = block_seconds
        self._state = CaptureState.IDLE
        self._stream = None
        self._mixer = AudioMixer()
        self._on_frame: Optional[Callable[[AudioFrame], None]] = None
        self._lock = threading.Lock()

    def permission_status(self) -> bool:
        return platform.system() == "Windows"

    def state(self) -> CaptureState:
        return self._state

    def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        if platform.system() != "Windows":
            raise CaptureDeviceError("WASAPI loopback is only available on Windows", source=self.kind)
        try:
            import sounddevice as sd
        except ImportError as e:
            raise CaptureDeviceError(
                "sounddevice/PortAudio not installed (pip install sounddevice)",
                source=self.kind,
            ) from e

        self._on_frame = on_frame
        try:
            device = self._device if self._device is not None else sd.default.device[1]
            info = sd.query_devices(device)
            native_rate = int(info["default_samplerate"])
            channels = int(info.get("max_output_channels", 0) or info.get("max_input_channels", 0) or 2)
            blocksize = int(self._block_seconds * native_rate)
            extra_settings = self._wasapi_loopback_settings(sd)
        except Exception as e:
            raise CaptureDeviceError(f"no WASAPI output device available: {e}", source=self.kind) from e

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
            if status and getattr(status, "input_overflow", False):
                _log.warning("WASAPI loopback overflow")
            try:
                frame = self._mixer.feed(
                    indata.copy(),
                    native_rate,
                    source=self.kind,
                    channels=channels,
                )
                if self._on_frame is not None:
                    self._on_frame(frame)
            except Exception:
                _log.exception("WASAPI loopback frame error")

        try:
            kwargs = {
                "device": device,
                "samplerate": native_rate,
                "channels": channels,
                "dtype": "float32",
                "blocksize": blocksize,
                "callback": _callback,
            }
            if extra_settings is not None:
                kwargs["extra_settings"] = extra_settings
            self._stream = sd.InputStream(**kwargs)
            self._stream.start()
            self._state = CaptureState.CAPTURING
            _log.info("WASAPI loopback capturing @ %d Hz, %d ch", native_rate, channels)
        except Exception as e:
            msg = str(e).lower()
            if "permission" in msg or "denied" in msg:
                self._state = CaptureState.PERMISSION_DENIED
                raise CapturePermissionError(str(e), source=self.kind) from e
            self._state = CaptureState.ERROR
            raise CaptureDeviceError(f"WASAPI loopback start failed: {e}", source=self.kind) from e

    @staticmethod
    def _wasapi_loopback_settings(sd):  # noqa: ANN001
        settings = getattr(sd, "WasapiSettings", None)
        if settings is None:
            return None
        try:
            return settings(loopback=True)
        except TypeError:
            return settings()

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                _log.exception("error closing WASAPI loopback stream")
        self._state = CaptureState.IDLE


__all__ = ["WasapiLoopbackCapture"]
