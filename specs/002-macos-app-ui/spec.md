# Feature Specification: macOS Meeting Assistant App — UI, End-to-End Validation & Packaging

**Feature Branch**: `002-macos-app-ui`  
**Created**: 2026-06-15  
**Status**: Draft  
**Input**: User description: "i want to create the ui and frontend design for this mac app and be able to test it and try it after you fully test that it is working correctly transcribing the voices, detect their languages, and separate between different voices. you can do this by testing with dump speeches you can download on internet or a small dataset on kaggle or huggingface for asr and speech diarization. make sure all are connected front + back and the app as a whole, easily to launch and well packaged"

## Clarifications

### Session 2026-06-15

- Q: How should the graphical front-end connect to the existing in-process Python backend? → A: A Python-based desktop app (native-feeling webview UI) calls the in-process Python backend directly with no cross-language bridge, packaged as a `.app`.
- Q: What packaging / distribution level is in scope for v1? → A: A locally double-click-launchable `.app` (ad-hoc signed) on the user's own Apple Silicon Mac; Developer-ID signing + Apple notarization are deferred to a later hardening pass.
- Q: What transcription-accuracy threshold defines "working correctly" on the curated public ASR samples? → A: Word Error Rate ≤ 15% on the curated clean speech samples.
- Q: Which transcript export format(s) must the app support? → A: Markdown (human-readable) and JSON (structured/machine-readable).
- Q: What is the target visual design ambition for v1? → A: A clean, polished, native-feeling macOS look (legible transcript, clear speaker color-coding, simple record/import controls, guided setup) — the "liquid glass" / live-waveform direction is a deferred enhancement.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run a live diarized, multilingual meeting transcription from the app (Priority: P1)

A user opens the packaged macOS app, starts a session, and watches their meeting be transcribed live: each utterance appears as a time-stamped line attributed to a speaker label (Speaker 1, 2, …) and tagged with the language spoken. Audio comes from both the local microphone and the meeting app's system audio, processed entirely on-device. When finished, the user stops the session and sees the complete transcript.

**Why this priority**: This is the headline experience and the smallest end-to-end slice that proves the whole product works — the UI is wired to the backend, real audio flows through capture → diarization → transcription → display. Everything else builds on it.

**Independent Test**: Launch the app, grant the requested permissions, click Start, speak (ideally with a second person or a meeting playing), and confirm live transcript lines appear with correct speaker labels, language tags, and timestamps; click Stop and confirm the full transcript is retained on screen.

**Acceptance Scenarios**:

1. **Given** the app is open and ready, **When** the user starts a session and speaks a sentence, **Then** within a few seconds a transcript line appears showing the recognized text, a speaker label, a language tag, and a timestamp.
2. **Given** a running session, **When** a second distinct voice speaks, **Then** the app shows that speech under a different, stable speaker label rather than merging it with the first speaker.
3. **Given** a meeting app is playing remote participants' audio while the user also speaks, **When** a session captures both sources, **Then** the on-screen transcript interleaves local and remote speakers on one time-ordered timeline.
4. **Given** a speaker talks in a different supported language, **When** their speech is transcribed, **Then** the line is shown in that language with the correct language tag.
5. **Given** a running session, **When** the user stops it, **Then** capture halts and the complete, ordered transcript remains visible for review.

---

### User Story 2 - Try and verify the app with a downloaded speech or sample dataset clip (Priority: P2)

Without setting up a live meeting, the user (or a tester) loads an existing audio file — a speech downloaded from the internet, or a clip from a public ASR/diarization dataset — and the app transcribes it into a diarized, language-tagged transcript. This makes the app easy to try on demand and is the practical way to confirm transcription, language detection, and speaker separation are working correctly on known content.

**Why this priority**: It removes the need for a live meeting to demonstrate or validate the app, and provides repeatable, inspectable inputs (where the expected speakers/languages/text are known) to confirm correctness.

**Independent Test**: From the app, choose an audio file containing one or more speakers and/or languages, run transcription, and confirm the produced transcript matches the known content — correct text, correct number of distinct speaker labels, and correct per-segment language tags.

**Acceptance Scenarios**:

1. **Given** a single-speaker audio file, **When** the user transcribes it, **Then** the transcript shows the spoken text under one speaker label with the correct language tag.
2. **Given** a multi-speaker audio file, **When** the user transcribes it, **Then** the app assigns a distinct, stable label to each distinct voice and attributes each segment to the right speaker.
3. **Given** an audio file containing more than one language, **When** the user transcribes it, **Then** each segment is transcribed in its spoken language and tagged with the identified language.
4. **Given** an unsupported file or unreadable audio, **When** the user selects it, **Then** the app shows a clear, actionable error instead of failing silently or hanging.

---

### User Story 3 - First-run setup and easy, well-packaged launch (Priority: P2)

A user receives the app as a self-contained macOS application, double-clicks it to launch (no terminal commands, no manual environment setup), and is guided through a one-time setup: required speech and diarization models are downloaded and cached, and the microphone and system-audio permissions are requested with plain-language explanations. The app clearly shows when it is ready to use, and on later launches it starts straight into a ready state without re-downloading.

**Why this priority**: "Easily to launch and well packaged" is an explicit requirement. A polished first-run flow is what turns the working pipeline into something a non-technical user can actually run, and it gates the P1 experience for anyone who hasn't already prepared models/permissions.

**Independent Test**: On a clean machine, double-click the packaged app, complete the guided setup once, and reach a "Ready" state; quit and relaunch, and confirm the app reaches "Ready" quickly from cache without re-downloading models.

**Acceptance Scenarios**:

1. **Given** a freshly installed app with no models cached, **When** the user launches it for the first time, **Then** the app guides model download with visible progress and reaches a clear "Ready" state on completion.
2. **Given** required permissions have not been granted, **When** the user reaches setup, **Then** the app explains why each permission is needed and lets the user grant microphone and system-audio access, reporting which are still missing.
3. **Given** models are already cached and permissions granted, **When** the user relaunches the app, **Then** it reaches "Ready" promptly without re-downloading.
4. **Given** model download is interrupted, **When** the user retries, **Then** setup resumes/retries cleanly without leaving the app in a broken state.

---

### User Story 4 - Review, navigate, and export the transcript (Priority: P3)

After (or during) a session, the user reviews the transcript in a readable layout — grouped or color-coded by speaker, with timestamps and language tags — and exports it to a common format (plain text, Markdown, or structured data) to save or share.

**Why this priority**: Capturing the transcript is only useful if the user can read it comfortably and get it out of the app. It depends on a working transcript existing (P1/P2) and is a natural follow-on rather than core proof of function.

**Independent Test**: Produce a transcript (live or from a file), confirm it is presented in a readable, speaker-attributed layout with timestamps and language tags, and export it; confirm the exported file contains the full transcript with speaker, time, language, and text.

**Acceptance Scenarios**:

1. **Given** a completed transcript, **When** the user views it, **Then** segments are visually attributed to speakers and ordered by time, with language tags shown.
2. **Given** a completed transcript, **When** the user exports it, **Then** a file is produced that preserves speaker labels, timestamps, languages, and text.
3. **Given** a long transcript, **When** the user scrolls or jumps to a speaker, **Then** the layout remains responsive and readable.

---

### User Story 5 - Accuracy validation against public sample datasets (Priority: P3)

A developer/tester runs a repeatable validation pass that feeds clips from public ASR, diarization, and multilingual datasets through the integrated app pipeline and produces a quality report — transcription accuracy, speaker-separation accuracy, and language-identification accuracy — so the team can confirm "it is working correctly" against objective, known-answer benchmarks before trusting the live experience.

**Why this priority**: This operationalizes the user's "fully test that it is working correctly" requirement with measurable evidence, but it is a QA/verification capability layered on top of the user-facing app rather than something an end user runs day-to-day.

**Independent Test**: Run the validation pass over a small curated set of public dataset clips with known transcripts/speaker labels/languages, and confirm it outputs accuracy metrics that meet the defined thresholds (see Success Criteria).

**Acceptance Scenarios**:

1. **Given** a curated set of labeled sample clips, **When** the validation pass runs, **Then** it reports transcription accuracy (e.g., word error rate), speaker-separation accuracy, and language-identification accuracy per clip and in aggregate.
2. **Given** the validation report, **When** results fall below defined thresholds, **Then** the failing cases are clearly identified for follow-up.
3. **Given** the same dataset and configuration, **When** the validation pass is re-run, **Then** results are reproducible within a small tolerance.

---

### Edge Cases

- **No models / not ready**: If a user tries to start a session before setup is complete, the app blocks with a clear prompt to finish setup rather than failing mid-capture.
- **Permission denied or revoked mid-session**: The app surfaces a clear state change (e.g., "microphone access lost") instead of silently producing nothing or hanging.
- **Silence / no speech**: The app does not fabricate text or spurious speaker labels during silence.
- **Many overlapping speakers**: When more voices speak than can be resolved, the app degrades gracefully and signals lower confidence rather than crashing.
- **Unsupported language**: Speech outside the supported set yields best-effort output flagged as low-confidence / unknown language rather than a hard failure.
- **Large or long audio file import**: Importing a long file shows progress and remains responsive rather than appearing frozen.
- **Backend not running / integration failure**: If the UI cannot reach the processing backend, it shows a clear error and recovery path instead of a blank or frozen screen.
- **Low compute / memory**: The app reports the constraint and degrades (buffering/lag signaling) rather than silently dropping audio.
- **App quit during an active session**: Stopping or quitting cleanly halts capture and does not leave orphaned processes or a corrupt transcript.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The app MUST provide a graphical macOS interface that lets a user start and stop a transcription session and see results, without using a terminal or writing code.
- **FR-002**: The app MUST display the live transcript as it is produced, showing for each segment the recognized text, a speaker label, a language tag, and a timestamp.
- **FR-003**: The app MUST visually distinguish different speakers (e.g., grouping, labeling, or color) using stable labels consistent for the duration of a session.
- **FR-004**: The app MUST support capturing both the local microphone and meeting-app system audio for a live session, and present both on one merged, time-ordered timeline.
- **FR-005**: The app MUST let a user select and transcribe an existing audio file (a downloaded speech or dataset clip) and display the resulting diarized, language-tagged transcript.
- **FR-006**: The app MUST integrate with the existing on-device Python transcription/diarization backend by calling it in-process (no cross-language bridge), so that all front-end actions are driven by real processing results (no mocked output in the shipped app).
- **FR-007**: The app MUST run all transcription, diarization, and language identification on-device; captured audio and transcripts MUST NOT be sent to external services for processing.
- **FR-008**: The app MUST provide a guided first-run setup that downloads and caches the required speech and diarization models with visible progress, and requests the necessary microphone and system-audio permissions with plain-language explanations.
- **FR-009**: The app MUST clearly indicate readiness state (e.g., setting up / ready / missing permissions / error) and what, if anything, the user must do to become ready.
- **FR-010**: On subsequent launches, the app MUST reach a ready state from cached models without re-downloading, and MUST handle interrupted downloads with a clean retry.
- **FR-011**: The app MUST be delivered as a self-contained, double-click-launchable macOS `.app` (ad-hoc signed, runnable on the user's own Apple Silicon Mac) that does not require the user to install developer tooling or run setup commands manually. Developer-ID signing and Apple notarization are out of scope for this feature.
- **FR-012**: The app MUST let the user review the completed transcript in a readable, speaker-attributed, time-ordered layout including language tags.
- **FR-013**: The app MUST let the user export a transcript to both Markdown (human-readable, speaker-grouped) and JSON (structured/machine-readable), each preserving speaker labels, timestamps, languages, and text.
- **FR-014**: The app MUST surface clear, actionable errors for failure conditions (missing permissions, missing models, unreadable audio, backend unavailable, lost audio device) instead of failing silently or hanging.
- **FR-015**: The app MUST handle sessions with two or more speakers and with more than one spoken language without forcing a single speaker or single language.
- **FR-016**: The system MUST provide a repeatable validation capability that runs public ASR/diarization/multilingual sample clips through the integrated pipeline and reports transcription, speaker-separation, and language-identification accuracy against known answers.
- **FR-017**: The validation capability MUST identify which sample cases pass or fail against defined accuracy thresholds and produce reproducible results across runs.
- **FR-018**: The app MUST remain responsive during long live sessions and long file imports, showing progress/activity rather than appearing frozen.
- **FR-019**: The app MUST preserve correct chronological ordering and coherent timestamps in the displayed and exported transcript even when processing lags behind input.
- **FR-020**: The app MUST allow a user to run multiple sessions over its lifetime, each with its own independent transcript and speaker labeling.

### Key Entities *(include if feature involves data)*

- **App Session**: A user-facing transcription run started from the UI. Has a status (setting up / ready / recording / processing / stopped / error), the selected input mode (live capture or file import), and the resulting transcript. Wraps the backend's transcription session.
- **Transcript (view model)**: The ordered collection of segments shown in the UI and exported, each carrying speaker label, start/end time, language tag, recognized text, and a confidence/quality indicator.
- **Speaker (view model)**: A distinct voice within a session, shown with a stable label and a consistent visual treatment; session-scoped and anonymous.
- **Audio Input Source**: Either live capture (microphone + system audio) or an imported audio file selected by the user.
- **Setup / Readiness State**: The app's view of model availability, permission grants, and compute capability, with a clear list of anything the user must resolve.
- **Export Artifact**: A saved file representing a transcript in a shareable format, preserving speaker, time, language, and text.
- **Validation Report**: The output of an accuracy validation pass over labeled sample clips — per-clip and aggregate transcription, diarization, and language-identification metrics, plus pass/fail against thresholds.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new user can go from double-clicking the app to a "Ready" state by completing only the guided on-screen setup (model download + permission grants), with no terminal use or manual configuration.
- **SC-002**: On a relaunch with models cached and permissions granted, the app reaches "Ready" in under 30 seconds.
- **SC-003**: During a live session, transcript lines for spoken phrases appear in the UI within ~2 seconds of the speech finishing (perceived near-real-time).
- **SC-004**: In a session with up to 4 distinct speakers, at least 90% of speech time is attributed to the correct, stable speaker label.
- **SC-005**: For clear speech in a supported language, the per-segment language is correctly identified for at least 95% of segments.
- **SC-006**: On the curated clean public ASR sample set, transcription accuracy is at or below a 15% word error rate (WER ≤ 15%), demonstrating the pipeline transcribes correctly.
- **SC-007**: The validation pass over public diarization and multilingual sample clips produces a report showing speaker-separation and language-identification accuracy at or above the SC-004/SC-005 thresholds, and the same run is reproducible within a small tolerance.
- **SC-008**: A user can import a downloaded speech or dataset clip and obtain a complete diarized, language-tagged transcript without any live audio setup.
- **SC-009**: A completed transcript can be exported to both Markdown and JSON files that preserve speaker, timestamp, language, and text, openable in standard tools.
- **SC-010**: All captured audio and transcripts remain on-device during normal operation — no audio or transcript content leaves the machine (verifiable by absence of outbound transfer of audio/transcript data).
- **SC-011**: The app sustains a continuous 60-minute live session without losing timeline continuity, dropping speaker labels, corrupting segment order, or becoming unresponsive.
- **SC-012**: A first-time user can complete the core flow (launch → ready → transcribe a sample clip → read result → export) in under 15 minutes using only the on-screen interface.

## Assumptions

- **Builds on the existing backend (001)**: This feature consumes the implemented on-device pipeline from `001-meeting-asr-backend` (capture → diarize → transcribe → fuse → session/facade). The 001 spec deferred all UI to "a later feature"; this is that feature.
- **Real models required for validation**: Confirming correct transcription, language detection, and speaker separation requires running the real speech (Nemotron ASR) and diarization (Sortformer) models on real audio. Completing real-model inference (a known gap from 001, which was offline-tested with fakes) is in scope for this feature so the app and validation produce genuine results.
- **Target platform**: macOS on Apple Silicon (M1+), matching the backend. System-audio capture requires the OS versions documented in 001 (Core Audio Process Taps on macOS 14.4+, ScreenCaptureKit fallback on 13.0–14.3).
- **Packaging scope (v1)**: "Well packaged, easily to launch" means a self-contained, double-click-launchable `.app` (ad-hoc signed) that runs on the user's own Apple Silicon Mac without installing developer tooling or running setup commands. Full Developer-ID code-signing and Apple notarization for frictionless distribution to *other* people's Macs are explicitly deferred to a later hardening step and out of scope for this feature.
- **Visual design ambition**: The UI is a clean, polished, native-feeling macOS desktop experience (clear transcript layout, speaker color-coding, readable language/timestamp metadata, an obvious record/import control, and a guided setup), consistent with the small-window/polished vision noted in 001. The more elaborate "liquid glass" aesthetic and live audio-waveform/visualization are deferred enhancements, not in scope for v1.
- **Validation datasets**: Correctness testing uses small, freely available, labeled sample clips from public ASR / speaker-diarization / multilingual sources (for example, the kinds of datasets on Hugging Face or Kaggle the user referenced). Exact dataset selection, licensing check, and the size of the curated subset are settled during planning; only a small representative subset is needed to demonstrate correctness, not full benchmark coverage.
- **Front↔back integration mechanism**: The front-end is a Python-based desktop app (native-feeling webview UI) that calls the existing in-process Python backend directly — no cross-language bridge or separate local server. This reuses the backend with the lowest integration risk and packages to a double-click `.app`. The app is fully connected end-to-end and ships no mocked output. The specific UI toolkit/packaging tooling is settled during planning.
- **Speaker identity is session-scoped and anonymous**: Speakers are labeled Speaker 1, 2, … per session; persistent voice enrollment / cross-meeting recognition is out of scope.
- **Summarization out of scope**: Producing meeting minutes / LLM summaries from the transcript remains a later concern; this feature stops at the diarized, multilingual transcript, its review/export, and validation.
- **Single-user, local app**: No accounts, multi-user sync, or cloud storage; transcripts and exports live on the local machine.
