# Contract: AudioCapture

**Module**: `meeting_asr.audio.capture` (protocol) → `coreaudio_tap`, `screencapturekit`,
`microphone` (backends), `mixer` (combiner).
**Constitution**: III (platform-native, behind a platform-agnostic interface), VII (modular).

```python
class AudioCapture(Protocol):
    kind: AudioSourceKind

    def permission_status(self) -> bool:
        """Whether capture is currently authorized for this source. No prompt side effects
        beyond what the OS requires."""

    def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        """Begin capture. Emits AudioFrame chunks (16 kHz mono float32) tagged with this
        source kind and session-clock timestamps. Raises CapturePermissionError or
        CaptureDeviceError with actionable hints (FR-015)."""

    def stop(self) -> None:
        """Stop capture and release the device/tap. Idempotent."""

    def state(self) -> CaptureState:
        ...
```

## Backends

- **`MicrophoneCapture`** — PortAudio via `sounddevice`; default input device; resample to 16 kHz mono.
- **`CoreAudioTapCapture`** (primary system audio, macOS ≥ 14.4) — spawns the `native/AudioTap`
  Swift helper (`AudioHardwareCreateProcessTap` + `CATapDescription`), reads raw PCM from its
  stdout pipe, resamples to 16 kHz mono. Meeting-app-agnostic (FR-002).
- **`ScreenCaptureKitCapture`** (fallback, macOS 13.0–14.3) — system audio via ScreenCaptureKit.

## Combiner

- **`AudioMixer`** — merges enabled sources onto one monotonic session clock, resamples each to
  16 kHz mono, and yields a single mixed `AudioFrame` stream to diarizer + ASR, while preserving
  per-frame `source` tags for best-effort origin labeling (FR-009).

## Contract rules

- Output is always **16 kHz mono float32** regardless of native device format.
- Timestamps are continuous and monotonic; gaps (silence/dropout) preserved, not collapsed.
- Permission/device changes mid-session transition `state()` and fire an error callback rather
  than hanging (edge cases: device unplugged, permission revoked).
- No frame is silently dropped under backpressure; the mixer buffers and signals lag (FR-016).
- All backends are testable from a recorded source: a tap dump file / fixture WAV can stand in
  for live capture (Constitution VII testability).
