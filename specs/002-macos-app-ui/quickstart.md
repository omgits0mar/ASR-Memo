# Quickstart — macOS Meeting Assistant App (002)

How to build, run, validate, and package the desktop app on Apple Silicon macOS.
Builds on the `001` backend (already implemented). Network is allowed only during
`make setup`, model download, and the one-time validation-sample fetch.

## Prerequisites

- macOS 14.4+ (Process Taps for system audio; 13.0–14.3 works mic-only via
  ScreenCaptureKit fallback), Apple Silicon (M1+).
- `brew install portaudio` (microphone capture).
- One-time deps: `make setup` then `make build-native` (Swift Process-Tap helper).

## 1. Run the app from source (dev loop)

```bash
make setup            # venv + deps (now includes pywebview)
make build-native     # system-audio helper (optional for mic-only)
make run              # launches the pywebview app (python -m app.main)
```

First launch → **guided setup** (US3):
1. The setup screen shows readiness (models / mic / system-audio) from
   `check_readiness()`, explaining anything missing in plain language.
2. Click **Download models** → progress bar driven by `prepare_progress` events;
   reaches **Ready** when cached. Interrupted downloads resume cleanly on retry.
3. Grant microphone (and, for remote participants, system-audio) permission when macOS
   prompts; the screen reports which are still missing.

## 2. Live diarized, multilingual session (US1)

1. From **Ready**, choose sources (Microphone, and/or System audio) and click **Start**.
2. Speak (ideally with a second voice or a meeting playing). Transcript lines appear
   within ~2 s, each with text, a **Speaker N** label (color-coded), a **language tag**,
   and a timestamp. A second voice gets a distinct stable label; mic + system audio
   interleave on one time-ordered timeline.
3. Click **Stop** → capture halts and the full transcript stays on screen for review.

## 3. Transcribe a downloaded clip / dataset sample (US2)

1. Click **Import audio…**, pick a speech file (a downloaded talk or an ASR/diarization
   dataset clip).
2. The app runs the same pipeline on the file with a progress indicator and shows the
   diarized, language-tagged transcript. Unreadable/unsupported files show a clear error.

This is the easiest way to *try* and *verify* the app without a live meeting.

## 4. Review & export (US4)

- The completed transcript is shown speaker-attributed, time-ordered, with language tags;
  long transcripts scroll responsively.
- Click **Export…** → choose **Markdown** (human-readable, speaker-grouped) or **JSON**
  (structured). The saved file preserves speaker, timestamp, language, and text.

## 5. Accuracy validation (US5 / "fully test that it works")

Run the repeatable, known-answer accuracy pass over the curated public clips:

```bash
make validate                         # python -m validation --axis all
# or target one axis:
python -m validation --axis asr       # WER on clean speech (target ≤ 15%)
python -m validation --axis diarization
python -m validation --axis language
python -m validation --report-json out/validation.json --report-md out/validation.md
```

The harness feeds each labeled clip through the **real integrated pipeline** and reports
per-clip + aggregate WER, diarization accuracy, and language-ID accuracy with pass/fail
vs. thresholds (WER ≤ 0.15, diarization ≥ 0.90, language-ID ≥ 0.95). Exit code is non-zero
if a threshold is missed. (Requires downloaded models on Apple Silicon.)

## 6. Build the double-click `.app` (US3 packaging / FR-011)

```bash
make app              # PyInstaller bundle + ad-hoc codesign → dist/MeetingAssistant.app
open dist/MeetingAssistant.app
```

The `.app` is self-contained and ad-hoc signed (no developer tooling needed by the user).
Models are **not** bundled — the in-app guided setup downloads them on first run and caches
them, so relaunch reaches **Ready** in under 30 s. (Developer-ID signing + notarization for
distribution to other Macs are deferred.)

## 7. Tests (offline, no models)

```bash
make test             # full offline suite (network guard) — incl. bridge/file/export/validation logic
make test-fast        # unit + contract only
```

Real-model accuracy gates and live-hardware paths are marked `needs_models` /
`needs_hardware` and skip cleanly offline; the bridge is exercised headlessly (no window).

## Acceptance smoke (matches Success Criteria)

| Step | Criterion |
|------|-----------|
| Double-click → guided setup → Ready (no terminal) | SC-001 |
| Relaunch reaches Ready < 30 s from cache | SC-002 |
| Live line appears ≤ ~2 s after speech | SC-003 |
| Up to 4 speakers, ≥90% correct attribution | SC-004 |
| ≥95% correct per-segment language | SC-005 |
| `make validate --axis asr` → WER ≤ 15% | SC-006 |
| `make validate` report reproducible, meets diar/lang thresholds | SC-007 |
| Import a clip → full diarized transcript | SC-008 |
| Export Markdown + JSON with all fields | SC-009 |
| No audio/transcript leaves the machine | SC-010 |
| 60-min live session stays coherent + responsive | SC-011 |
| First-timer completes launch→export < 15 min | SC-012 |
