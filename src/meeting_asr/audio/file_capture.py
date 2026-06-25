"""FileCapture — file → 16 kHz mono AudioFrames (task T026 / US2; research Decision 3).

Production promotion of the test ``FixtureCapture``: read any ``soundfile``-decodable
file, downmix to mono, resample to **16 kHz** via the shared mixer helpers, and emit
:class:`AudioFrame`s on the session clock with continuous timestamps. Implements
:class:`AudioCapture` (Constitution III/VII) so the file path reuses the whole
downstream pipeline — the result is identical whether audio arrives live or from a
file. Reports a 0..1 progress fraction derived from samples consumed.

Errors surface as :class:`FileCaptureError` (a :class:`CaptureDeviceError`) carrying
a structured :class:`ErrorInfo` (``audio.unreadable`` / ``audio.empty``) instead of
hanging (FR-014, US2 scenario 4).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .._logging import CaptureDeviceError, get_logger
from ..types import AudioSourceKind, CaptureState, ErrorInfo
from .mixer import SAMPLE_RATE, AudioMixer, resample, to_mono

_log = get_logger("audio.file_capture")

__all__ = ["FileCapture", "FileCaptureError"]


class FileCaptureError(CaptureDeviceError):
    """A file could not be read/decoded; carries an actionable :class:`ErrorInfo`."""

    def __init__(self, code: str, message: str, *, hint: Optional[str] = None) -> None:
        super().__init__(message, source=AudioSourceKind.MICROPHONE)
        self.code = code
        self.info = ErrorInfo(code=code, message=message, recoverable=True, hint=hint)


class FileCapture:
    """AudioCapture-conforming file reader: streams a file as 16 kHz mono frames."""

    kind: AudioSourceKind = AudioSourceKind.MICROPHONE

    def __init__(self, path: str, *, block_seconds: float = 0.1, realtime: bool = False) -> None:
        self.kind = AudioSourceKind.MICROPHONE  # files are a mic-equivalent source
        self._path = str(path)
        self._block_n = max(1, int(block_seconds * SAMPLE_RATE))
        self._realtime = realtime
        self._pcm = self._decode(self._path)  # eager: surfaces errors up front (FR-014)
        self._state = CaptureState.IDLE
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mixer = AudioMixer()
        self._consumed = 0  # samples emitted so far

    # ---- decoding ----

    @staticmethod
    def _decode(path: str) -> np.ndarray:
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise FileCaptureError(
                "audio.unreadable", f"file not found: {path}",
                hint="Pick an existing audio file (WAV/FLAC/MP3).",
            )
        try:
            import soundfile as sf

            pcm, sr = sf.read(str(p), dtype="float32", always_2d=False)
        except Exception as e:  # unsupported/undecodable format
            raise FileCaptureError(
                "audio.unreadable", f"could not decode {Path(path).name}: {e}",
                hint="Use a WAV/FLAC/MP3/Ogg speech file.",
            ) from e
        pcm = np.asarray(pcm)
        if pcm.size == 0 or (pcm.ndim > 1 and pcm.shape[0] == 0):
            raise FileCaptureError(
                "audio.empty", f"empty / zero-length audio: {Path(path).name}",
                hint="Choose a file that actually contains speech.",
            )
        mono = to_mono(pcm)
        if sr != SAMPLE_RATE:
            mono = resample(mono, sr, SAMPLE_RATE)
        return np.ascontiguousarray(mono, dtype=np.float32)

    # ---- AudioCapture protocol ----

    def permission_status(self) -> bool:
        return True  # files need no OS permission

    def state(self) -> CaptureState:
        return self._state

    def start(self, on_frame: Callable) -> None:
        self._state = CaptureState.CAPTURING
        self._stop.clear()
        self._consumed = 0

        def _run() -> None:
            n = len(self._pcm)
            i = 0
            try:
                while i < n and not self._stop.is_set():
                    block = self._pcm[i : i + self._block_n]
                    if block.size == 0:
                        break
                    # Account for the block being emitted so progress reflects it.
                    self._consumed = min(i + len(block), n)
                    frame = self._mixer.feed(block, SAMPLE_RATE, source=self.kind)
                    on_frame(frame)
                    i += self._block_n
                    if self._realtime:
                        time.sleep(self._block_n / SAMPLE_RATE)
            except Exception:  # never let a capture-thread error strand silently
                _log.exception("FileCapture streaming error")
            finally:
                if not self._stop.is_set():
                    self._state = CaptureState.IDLE

        self._thread = threading.Thread(target=_run, name="FileCapture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Idempotent stop; joins the streaming thread."""
        self._stop.set()
        if self._state == CaptureState.CAPTURING:
            self._state = CaptureState.IDLE
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # ---- progress (US2) ----

    def total_seconds(self) -> float:
        """Decoded duration at the pipeline rate."""
        return len(self._pcm) / SAMPLE_RATE

    def consumed_fraction(self) -> float:
        """Monotonic 0..1 fraction of the file streamed so far."""
        n = len(self._pcm)
        return (self._consumed / n) if n else 0.0
