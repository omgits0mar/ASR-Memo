"""System readiness assembly (tasks T040; FR-013, FR-015).

Builds :class:`SystemReadinessReport` from: model cache states, mic + system-audio
permission probes, the resolved compute backend, and OS system-audio capability.
Never raises for "not ready" — it enumerates what's missing.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

from .._logging import get_logger
from ..backends.device import DeviceProbe, resolve_backend
from ..types import SystemReadinessReport
from .registry import default_cache_dir, model_registry, refresh_state

_log = get_logger("models.readiness")

_MIN_TAP_MAJOR, _MIN_TAP_MINOR = 14, 4  # Process Taps require macOS 14.4+


def os_supports_process_tap() -> bool:
    """True iff the OS is macOS ≥ 14.4 (Core Audio Process Taps)."""
    if platform.system() != "Darwin":
        return False
    try:
        release = platform.mac_ver()[0]
        major, minor = (int(x) for x in release.split(".")[:2])
    except (ValueError, IndexError):
        return False
    return (major, minor) >= (_MIN_TAP_MAJOR, _MIN_TAP_MINOR)


def _linux_has_monitor_source() -> bool:
    """Best-effort PipeWire/PulseAudio monitor-source detection."""
    try:
        import sounddevice as sd  # type: ignore

        devices = sd.query_devices()
    except Exception:
        return False
    for dev in devices:
        name = str(dev.get("name", "")).lower()
        if int(dev.get("max_input_channels", 0) or 0) > 0 and (
            ".monitor" in name or "monitor of" in name
        ):
            return True
    return False


def os_supports_system_audio() -> bool:
    """True iff this OS has a supported system-audio loopback path."""
    system = platform.system()
    if system == "Darwin":
        return os_supports_process_tap()
    if system == "Windows":
        return True
    if system == "Linux":
        return _linux_has_monitor_source()
    return False


def mic_permission() -> bool:
    """Best-effort microphone availability (default input device present).

    macOS TCC permission cannot be queried without prompting; a denial surfaces
    as ``CapturePermissionError`` on capture start.
    """
    try:
        import sounddevice as sd  # type: ignore

        return sd.default.device[0] is not None
    except Exception:
        return False


def system_audio_permission() -> bool:
    """Best-effort system-audio capture capability.

    macOS TCC prompts on the Swift helper's first run; Windows/Linux device errors
    surface from their capture backends.
    """
    return os_supports_system_audio()


def build_readiness(
    *,
    cache_root: Optional[Path] = None,
    probe: Optional[DeviceProbe] = None,
) -> SystemReadinessReport:
    """Assemble the full readiness snapshot (FR-013). Never raises."""
    root = Path(cache_root or default_cache_dir())
    backend = resolve_backend(probe)
    models = [refresh_state(a, root) for a in model_registry(backend)]
    mic = mic_permission()
    sys_audio = system_audio_permission()
    supports_system = os_supports_system_audio()

    missing: list[str] = []
    for a in models:
        if not a.is_cached():
            missing.append(f"model '{a.name}' not downloaded — run prepare_models()")
    if not mic:
        missing.append("microphone not available/authorized (grant in System Settings → Privacy → Microphone)")
    # System audio is only required when a SYSTEM source is used; report as advisory.
    if not sys_audio:
        missing.append(
            "system-audio loopback unavailable — required only for remote participants"
        )

    return SystemReadinessReport(
        models=models,
        mic_permission=mic,
        system_audio_permission=sys_audio,
        compute_backend=backend.value,
        os_supports_system_audio=supports_system,
        os_supports_process_tap=os_supports_process_tap(),
        missing=missing,
    )


__all__ = [
    "build_readiness",
    "os_supports_process_tap",
    "os_supports_system_audio",
    "mic_permission",
    "system_audio_permission",
]
