from __future__ import annotations

from meeting_asr.audio.pipewire_loopback import PipeWireLoopbackCapture
from meeting_asr.audio.wasapi_loopback import WasapiLoopbackCapture


class FakeSoundDevice:
    @staticmethod
    def query_devices():
        return [
            {"name": "Built-in Microphone", "max_input_channels": 1},
            {"name": "alsa_output.pci.stereo.monitor", "max_input_channels": 2},
        ]


def test_pipewire_monitor_device_discovery():
    assert PipeWireLoopbackCapture._find_monitor_device(FakeSoundDevice) == 1


def test_pipewire_monitor_device_absent():
    class NoMonitor:
        @staticmethod
        def query_devices():
            return [{"name": "Built-in Microphone", "max_input_channels": 1}]

    assert PipeWireLoopbackCapture._find_monitor_device(NoMonitor) is None


def test_wasapi_loopback_settings_uses_loopback_flag_when_available():
    class SD:
        class WasapiSettings:
            def __init__(self, *, loopback=False):
                self.loopback = loopback

    settings = WasapiLoopbackCapture._wasapi_loopback_settings(SD)
    assert settings.loopback is True
