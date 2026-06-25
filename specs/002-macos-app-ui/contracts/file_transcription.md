# Contract — File-Import Capture & Transcription

Adds file input (US2) by promoting the test `FixtureCapture` to a production
`AudioCapture` backend and exposing a facade entry point. No pipeline change.

## `FileCapture` (implements `AudioCapture`)

Module: `src/meeting_asr/audio/file_capture.py`.

```python
class FileCapture:                      # conforms to meeting_asr.audio.capture.AudioCapture
    kind: AudioSourceKind = AudioSourceKind.MICROPHONE   # files are treated as a mic-equivalent source
    def __init__(self, path: str, *, block_seconds: float = 0.1,
                 realtime: bool = False) -> None: ...
    def permission_status(self) -> bool: ...   # True (no OS permission for files)
    def start(self, on_frame: Callable[[AudioFrame], None]) -> None: ...
    def stop(self) -> None: ...                # idempotent
    def state(self) -> CaptureState: ...
    # extras for progress reporting:
    def total_seconds(self) -> float: ...      # decoded duration
    def consumed_fraction(self) -> float: ...  # 0..1 progress
```

**Behavior**:
- Reads any `soundfile`-decodable file; downmix to mono; resample to **16 kHz** via `soxr`
  (reusing the `AudioMixer.feed` path), emitting `AudioFrame`s on the session clock with
  continuous timestamps (no silent drops).
- `start` runs a worker thread that streams blocks until EOF, then signals end. With
  `realtime=False` (default) it streams as fast as the pipeline accepts; `realtime=True`
  paces to wall-clock (for demo).
- **Errors** (raise `CaptureDeviceError`/structured `ErrorInfo` with actionable hint, not
  hang — FR-014, US2 sc.4):
  - missing/unreadable/unsupported file → `code="audio.unreadable"`
  - empty / zero-length audio → `code="audio.empty"`

## Facade entry point

Module: `src/meeting_asr/__init__.py` — new public function (re-exported in `__all__`):

```python
def transcribe_file(
    path: str,
    *,
    language_hint: Optional[str] = None,
    on_segment: Optional[Callable[[TranscriptSegment], None]] = None,
    on_error: Optional[Callable[[ErrorInfo], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    _backends: Optional[Backends] = None,
) -> TranscriptionSession: ...
```

**Contract**:
- Builds a `FileCapture(path)` + the default diarizer/transcriber (or injected `_backends`),
  wires the existing `Pipeline`/`TranscriptionSession`, loads models, and runs the file to
  completion; returns the session in `STOPPED` (transcript fully available) — OR returns a
  session in `ERROR` with `ErrorInfo` for unreadable input.
- Honors the same readiness/permission/busy rules as `start_session`
  (`ReadinessError` when models missing; `SessionBusyError` when one is active — FR-020).
- `on_segment` fires per finalized segment exactly as live capture does (same downstream
  fusion → identical result whether live or from file).
- `on_progress` reports a monotonic 0..1 fraction derived from
  `FileCapture.consumed_fraction()`; reaches `1.0` at completion.
- `language_hint=None` ⇒ per-turn auto-detection (Principle VI); detected language recorded
  per segment.

## Testability

- `tests/unit/test_file_capture.py`: WAV → frames count/duration, resample correctness,
  16 kHz mono output, unreadable/empty/short-file error codes, idempotent `stop`,
  `consumed_fraction` monotonicity.
- `tests/contract/test_file_transcription.py`: `transcribe_file` with injected fakes
  produces an ordered transcript, fires `on_progress` to 1.0, surfaces `not_ready`/`busy`,
  and returns `ERROR` for a bad path — all offline.
- `needs_models`: real-model file transcription on a fixture clip (feeds the validation
  harness).
