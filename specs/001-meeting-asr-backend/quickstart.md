# Quickstart: Realtime Diarized Meeting Transcription Backend

**Feature**: `001-meeting-asr-backend` | macOS 14.4+ · Apple Silicon

A backend-only, in-process Python library. No UI. Drives the full lifecycle:
prepare models → check readiness → start → receive segments → stop. Target: ~1.5–2.0s
turn-to-text on Apple Silicon.

## 1. Environment setup (one-time, network allowed)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # onnxruntime, coremltools, huggingface_hub,
                                        # sounddevice, numpy, soxr, soundfile, ...
brew install portaudio                  # PortAudio for mic capture
swift build -c release --package-path native/AudioTap   # build the Core Audio Process-Tap helper
```

## 2. Grant macOS permissions

- **Microphone**: System Settings → Privacy & Security → Microphone → enable for your terminal/app.
- **System audio (Process Tap)**: approve the capture prompt on first run (macOS 14.4+).
  `check_readiness()` reports what is still missing.

## 3. Download / cache the models (one API call)

```python
from meeting_asr import prepare_models

report = prepare_models(progress=lambda p: print(p.asset, p.state, f"{p.downloaded}/{p.total}"))
assert report.ready, report.missing
# Downloads Nemotron 3.5 ASR (FP16 ONNX) + Streaming Sortformer (CoreML), then caches.
# Subsequent runs load from cache in <30s with no network (SC-005).
```

## 4. Run a live diarized session

```python
from meeting_asr import start_session, check_readiness, AudioSourceKind

print(check_readiness())   # models, mic/system permissions, compute backend ("coreml-gpu+cpu" / "cpu")

session = start_session(
    sources=(AudioSourceKind.MICROPHONE, AudioSourceKind.SYSTEM),  # you + meeting app
    language_hint=None,                                            # auto per-turn language
    on_segment=lambda s: print(f"[{s.start:6.1f}-{s.end:6.1f}] {s.speaker_label} "
                               f"({s.language}): {s.text}"),
)

# ... join your Teams/Meet/Zoom call; speak; let others speak ...

import time; time.sleep(60)
final = session.stop()     # returns the complete transcript
print(f"{len(final)} segments, speakers: {list(session.speakers())}")
```

Equivalent pull-style consumption:

```python
session = start_session()
for seg in session.segments():        # blocking iterator of finalized segments
    print(seg.speaker_label, seg.text)
```

Query the live transcript any time while ACTIVE (FR-019): `session.transcript()`.

## 5. Validation checklist (maps to spec)

- [ ] **US1/SC-001**: spoken phrase appears as a labeled, timestamped segment within ~2s.
- [ ] **US1**: a second voice gets a distinct, stable "Speaker 2" label.
- [ ] **US2/SC-004**: with a meeting app playing + mic live, both local and remote voices appear
      on one merged timeline.
- [ ] **US2/FR-015**: revoking system-audio permission yields a clear, actionable error.
- [ ] **US3/SC-003**: two languages across speakers → correct per-segment `language` tags.
- [ ] **US4/SC-005**: second startup reaches "ready" from cache in <30s, no download.
- [ ] **SC-008**: a 60-min session keeps ordering/labels intact.

## 6. Testing (offline)

```bash
pytest tests/unit tests/contract           # logic + protocol conformance
pytest tests/integration                   # full pipeline on tests/fixtures/audio/ (no live capture)
```
Fixtures stand in for live capture; CI runs fully offline after setup (Constitution gates).

## Notes

- All processing is on-device; only `prepare_models()` touches the network. Audio/transcripts
  never leave the host (SC-006).
- The existing `transcribe_meeting.py` (Whisper batch prototype) is superseded by this library.
- Diarization (Sortformer) and ASR (Nemotron) run **in parallel** and are fused by timestamp for
  lowest latency — see `plan.md` Complexity Tracking.
