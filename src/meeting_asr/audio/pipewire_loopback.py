"""Linux system-audio capture via PulseAudio/PipeWire monitor sources."""

from __future__ import annotations

import platform
import threading
from typing import Callable, Optional

import numpy as np

from .._logging import CaptureDeviceError, CapturePermissionError, get_logger
from ..types import AudioFrame, AudioSourceKind, CaptureState
from .mixer import AudioMixer

_log = get_logger("audio.pipewire_loopback")


class PipeWireLoopbackCapture:
    """AudioCapture-conforming Linux monitor-source backend."""

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
        if platform.system() != "Linux":
            return False
        try:
            import sounddevice as sd

            return self._find_monitor_device(sd) is not None
        except Exception:
            return False

    def state(self) -> CaptureState:
        return self._state

    def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        if platform.system() != "Linux":
            raise CaptureDeviceError(
                "PipeWire/PulseAudio monitor capture is only available on Linux",
                source=self.kind,
            )
        try:
            import sounddevice as sd
        except ImportError as e:
            raise CaptureDeviceError(
                "sounddevice/PortAudio not installed (pip install sounddevice)",
                source=self.kind,
            ) from e

        self._on_frame = on_frame
        try:
            device = self._device if self._device is not None else self._find_monitor_device(sd)
            if device is None:
                raise RuntimeError("no .monitor input source found")
            info = sd.query_devices(device)
            native_rate = int(info["default_samplerate"])
            channels = int(info.get("max_input_channels", 0) or 2)
            blocksize = int(self._block_seconds * native_rate)
        except Exception as e:
            raise CaptureDeviceError(f"no PipeWire/PulseAudio monitor source available: {e}", source=self.kind) from e

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
            if status and getattr(status, "input_overflow", False):
                _log.warning("PipeWire monitor overflow")
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
                _log.exception("PipeWire monitor frame error")

        try:
            self._stream = sd.InputStream(
                device=device,
                samplerate=native_rate,
                channels=channels,
                dtype="float32",
                blocksize=blocksize,
                callback=_callback,
            )
            self._stream.start()
            self._state = CaptureState.CAPTURING
            _log.info("PipeWire monitor capturing @ %d Hz, %d ch", native_rate, channels)
        except Exception as e:
            msg = str(e).lower()
            if "permission" in msg or "denied" in msg:
                self._state = CaptureState.PERMISSION_DENIED
                raise CapturePermissionError(str(e), source=self.kind) from e
            self._state = CaptureState.ERROR
            raise CaptureDeviceError(f"PipeWire monitor start failed: {e}", source=self.kind) from e

    @staticmethod
    def _find_monitor_device(sd) -> Optional[int]:  # noqa: ANN001
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            name = str(dev.get("name", "")).lower()
            if int(dev.get("max_input_channels", 0) or 0) > 0 and (
                ".monitor" in name or "monitor of" in name
            ):
                return idx
        return None

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                _log.exception("error closing PipeWire monitor stream")
        self._state = CaptureState.IDLE


__all__ = ["PipeWireLoopbackCapture"]
