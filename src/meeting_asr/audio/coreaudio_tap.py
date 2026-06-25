"""CoreAudioTapCapture — system audio via the Swift Process-Tap helper (task T029).

Primary system-audio backend (macOS ≥ 14.4). Spawns the bundled
``native/AudioTap`` Swift helper (Core Audio Process Taps), reads raw float32 PCM
from its stdout pipe, resamples to 16 kHz mono, and emits ``AudioFrame``s tagged
``source=SYSTEM``. Meeting-app-agnostic (FR-002). Permission denial / device loss
surfaces as ``CapturePermissionError`` / ``CaptureDeviceError`` (FR-015).

Testable offline: :meth:`start` accepts a ``helper_path`` that may point at any
executable emitting raw float32 PCM on stdout (e.g. a recorded tap dump replay
script — see ``tests/integration/test_audiotap_helper.py``).
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .._logging import CaptureDeviceError, CapturePermissionError, get_logger
from ..models.readiness import os_supports_process_tap
from ..types import AudioFrame, AudioSourceKind, CaptureState
from .mixer import SAMPLE_RATE, AudioMixer

_log = get_logger("audio.coreaudio_tap")


def _default_helper_path() -> Optional[Path]:
    """Resolve the built AudioTap helper, if present (dev repo or packaged .app)."""
    # Packaged .app: the spec bundles the helper at the PyInstaller _MEIPASS root.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "AudioTap"
        if bundled.exists():
            return bundled
    # Dev repo: <pkg>/src/meeting_asr/audio/coreaudio_tap.py → repo root/native/AudioTap
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "native" / "AudioTap" / ".build" / "release" / "AudioTap"
    if candidate.exists():
        return candidate
    return None


class CoreAudioTapCapture:
    """AudioCapture-conforming system-audio backend (Swift Process-Tap helper)."""

    kind = AudioSourceKind.SYSTEM

    def __init__(
        self,
        *,
        helper_path: Optional[str] = None,
        process: Optional[str] = None,
        block_seconds: float = 0.1,
    ) -> None:
        self._explicit_helper = helper_path is not None
        self._helper = helper_path or (_default_helper_path() and str(_default_helper_path()))
        self._process = process  # None → system-wide tap
        self._block = int(block_seconds * SAMPLE_RATE)
        self._state = CaptureState.IDLE
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._native_rate = SAMPLE_RATE
        self._stderr_tail = b""
        self._mixer = AudioMixer()

    def permission_status(self) -> bool:
        return self._can_replay_helper_off_platform() or os_supports_process_tap()

    def state(self) -> CaptureState:
        return self._state

    def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        if not self._can_replay_helper_off_platform() and not os_supports_process_tap():
            raise CapturePermissionError(
                "Core Audio Process Taps require macOS 14.4+", source=self.kind
            )
        if not self._helper:
            raise CaptureDeviceError(
                "AudioTap helper not found — run `swift build -c release --package-path native/AudioTap`",
                source=self.kind,
            )

        cmd = [self._helper]
        if self._process:
            cmd += ["--process", self._process]
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - controlled local helper
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as e:
            raise CaptureDeviceError(f"AudioTap helper not executable: {self._helper}", source=self.kind) from e

        # The helper emits at the device-native rate (≈48 kHz). It announces it on
        # stderr as `rate=<N>` before the first PCM block; drain stderr on a side
        # thread so we both pick up the rate and surface a permission denial that
        # the helper writes there (without it, a synchronous stderr.read() would
        # only run after stdout closed).
        self._native_rate = SAMPLE_RATE  # default until the helper reports its rate
        self._stderr_tail = b""
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="CoreAudioTap-stderr", daemon=True
        )
        self._stderr_thread.start()

        self._state = CaptureState.CAPTURING
        self._thread = threading.Thread(target=self._read_loop, args=(on_frame,), name="CoreAudioTap", daemon=True)
        self._thread.start()
        _log.info("system-audio tap started via %s", self._helper)

    def _drain_stderr(self) -> None:
        """Read the helper's stderr line by line: capture the announced native
        rate (`rate=<N>`) and remember the tail for permission-denial detection."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for raw_line in iter(proc.stderr.readline, b""):
                self._stderr_tail = raw_line
                line = raw_line.decode(errors="replace").strip()
                if line.startswith("rate="):
                    try:
                        rate = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    if rate > 0:
                        self._native_rate = rate
                        _log.info("system-audio tap native rate: %d Hz", rate)
                if "permission" in line.lower() or "not authorized" in line.lower():
                    self._state = CaptureState.PERMISSION_DENIED
                    _log.error("system-audio permission denied by helper: %s", line)
        except Exception:
            _log.exception("CoreAudioTap stderr drain error")

    def _can_replay_helper_off_platform(self) -> bool:
        """Allow local fake helpers to exercise the pipe reader on non-macOS CI."""
        if not self._explicit_helper or not self._helper:
            return False
        try:
            path = Path(self._helper).resolve()
        except OSError:
            return False
        system_roots = (Path("/bin"), Path("/sbin"), Path("/usr/bin"), Path("/usr/sbin"))
        return not any(path == root or root in path.parents for root in system_roots)

    def _read_loop(self, on_frame: Callable[[AudioFrame], None]) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        bytes_per_block = self._block * 4  # float32
        try:
            while True:
                raw = self._proc.stdout.read(bytes_per_block)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype=np.float32)
                if samples.size == 0:
                    continue
                # Use the helper's announced native rate (≈48 kHz); the mixer
                # resamples to 16 kHz. Falls back to SAMPLE_RATE until reported.
                frame = self._mixer.feed(samples, self._native_rate, source=self.kind)
                on_frame(frame)
        except Exception:
            _log.exception("CoreAudioTap read loop error")
        finally:
            if self._stderr_thread and self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=1.0)
            tail = self._stderr_tail or b""
            if (
                self._state is CaptureState.PERMISSION_DENIED
                or b"permission" in tail.lower()
                or b"not authorized" in tail.lower()
            ):
                self._state = CaptureState.PERMISSION_DENIED
                _log.error("system-audio permission denied by helper: %s", tail.decode(errors="replace"))
            else:
                self._state = CaptureState.IDLE

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1.0)
        if self._state is not CaptureState.PERMISSION_DENIED:
            self._state = CaptureState.IDLE


__all__ = ["CoreAudioTapCapture"]
