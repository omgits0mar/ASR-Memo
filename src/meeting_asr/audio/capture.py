"""AudioCapture protocol + AudioFrame (task T007).

Constitution III (platform-native behind a platform-agnostic interface) + VII
(modular). Concrete backends: ``microphone``, ``coreaudio_tap``,
``screencapturekit``; combiner: ``mixer``.

Output is always **16 kHz mono float32** regardless of native device format, with
continuous monotonic session-clock timestamps. No frame is silently dropped under
backpressure (FR-016); the mixer buffers and signals lag instead.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from .._logging import CaptureDeviceError, CapturePermissionError
from ..types import AudioFrame, AudioSourceKind, CaptureState

# Re-export AudioFrame so `from meeting_asr.audio.capture import AudioFrame` works
# as documented in the contract (canonical definition lives in ``types``).
__all__ = ["AudioCapture", "AudioFrame", "CaptureState", "CapturePermissionError", "CaptureDeviceError"]


@runtime_checkable
class AudioCapture(Protocol):
    """Platform-agnostic audio capture interface.

    ``start`` emits ``AudioFrame`` chunks (16 kHz mono float32) tagged with this
    source kind and session-clock timestamps. Raises ``CapturePermissionError`` or
    ``CaptureDeviceError`` (with actionable hints) on failure (FR-015).
    """

    kind: AudioSourceKind

    def permission_status(self) -> bool:
        """Whether capture is currently authorized. No prompt side effects beyond
        what the OS requires."""
        ...

    def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        """Begin capture. Emits 16 kHz mono float32 frames. Raises on permission/
        device failure."""
        ...

    def stop(self) -> None:
        """Stop capture and release the device/tap. Idempotent."""
        ...

    def state(self) -> CaptureState:
        """Current capture state."""
        ...
