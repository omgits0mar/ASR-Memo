# Feature Specification: Realtime Diarized Meeting Transcription Backend

**Feature Branch**: `001-meeting-asr-backend`  
**Created**: 2026-06-14  
**Status**: Draft  
**Input**: User description: "a small app window ... uses the mac audio apis to listen to my voice from mic and the people talking to me in any meeting app (Microsoft Teams, Google Meet, ...). Separate the voices via speaker diarization (can be more than 2 voices; identify them as Speaker 1, 2, ...) then collect and transcribe. No front-end for now — backend and APIs only. STT model: NVIDIA Nemotron 3.5 ASR (multilingual, streaming). Explore latest research for speaker diarization. Focus on model download, usage, core implementation, architecture, main methods, and exposing APIs. Prepares for a later UI feature."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Live diarized transcription of a single audio source (Priority: P1)

A user is in a meeting and wants their own spoken words captured and transcribed in real time, with the running transcript broken into time-stamped segments and attributed to a speaker label. The backend captures the user's microphone, continuously separates who is speaking, and emits transcript segments as the conversation unfolds.

**Why this priority**: This is the smallest end-to-end slice that proves the core pipeline (capture → diarize → transcribe → emit). It delivers a usable live transcript on its own and de-risks every downstream story.

**Independent Test**: Start a transcription session against the local microphone, speak (optionally with a second person on the same mic), and confirm the backend streams out transcript segments, each carrying a speaker label, start/end time, and recognized text — without any UI.

**Acceptance Scenarios**:

1. **Given** a started session bound to the microphone, **When** the user speaks a sentence, **Then** within a short delay the backend emits a transcript segment containing the recognized text, a speaker label (e.g., "Speaker 1"), and start/end timestamps.
2. **Given** an ongoing session, **When** a second distinct voice speaks on the same input, **Then** the backend attributes that speech to a different speaker label (e.g., "Speaker 2") rather than merging it with Speaker 1.
3. **Given** an active session, **When** the consumer requests the current accumulated transcript, **Then** the backend returns all segments so far in chronological order with stable speaker labels.
4. **Given** an active session, **When** the consumer stops the session, **Then** capture and processing halt and a final, complete transcript for the session is available.

---

### User Story 2 - Capture remote meeting participants alongside the local speaker (Priority: P2)

The user is on a call in a meeting app (Microsoft Teams, Google Meet, Zoom, etc.). They want not only their own mic captured but also the voices of the other participants coming out of the system audio, all merged into one diarized timeline so every speaker in the meeting is separated and labeled.

**Why this priority**: Capturing the far-end participants is what makes this a *meeting* assistant rather than a dictation tool. It depends on the P1 pipeline already working and adds the second, harder audio source.

**Independent Test**: With a meeting app playing audio (or a stand-in audio playback) plus live microphone input, start a session that captures both sources and confirm the merged transcript contains segments attributed to the local speaker and to one or more remote speakers on a single shared timeline.

**Acceptance Scenarios**:

1. **Given** the meeting-app audio is playing through system output and the user is also speaking into the mic, **When** a session capturing both sources is running, **Then** the backend produces a single time-ordered transcript that includes both the local speaker and the remote participants.
2. **Given** three or more distinct voices across mic and system audio, **When** they speak during the session, **Then** the backend assigns each a distinct, stable speaker label for the duration of the session.
3. **Given** the local speaker and a remote participant talk over each other briefly, **When** the overlap occurs, **Then** the backend still attributes speech to the correct speakers rather than dropping a speaker or collapsing both into one label.
4. **Given** the required system-audio capture permission has not been granted, **When** a session that needs system audio is started, **Then** the backend reports a clear, actionable permission error instead of silently capturing nothing.

---

### User Story 3 - Multilingual transcription with per-segment language identification (Priority: P3)

Participants may speak different languages, or one speaker may switch languages mid-meeting. The user wants each segment transcribed in the language actually spoken, with the detected language recorded on the segment.

**Why this priority**: Multilingual support is an explicit requirement and a differentiator, but it builds on top of a working single-language pipeline and can be layered in after the core capture/diarize/transcribe loop works.

**Independent Test**: Feed audio containing more than one language (across speakers or within one speaker) and confirm each emitted segment carries the correct recognized text in its spoken language and an identified language tag, drawing from the supported language set.

**Acceptance Scenarios**:

1. **Given** a session, **When** a speaker talks in one of the supported languages, **Then** the emitted segment contains text in that language and a language identifier for the segment.
2. **Given** two speakers using two different supported languages, **When** they each speak, **Then** each speaker's segments are transcribed in their respective language without forcing both into a single language.
3. **Given** a consumer optionally specifies an expected language for a session, **When** transcription runs, **Then** the backend uses that hint to bias recognition while still producing per-segment language tags.

---

### User Story 4 - Programmatic model lifecycle and session control API (Priority: P3)

A developer building the future UI (or an automated test) needs a clean programmatic surface to manage the speech models (ensure they are downloaded and loaded) and to drive transcription sessions (start, stream results, query, stop) plus check system readiness — all without any graphical interface.

**Why this priority**: The exposed API *is* the deliverable for this backend-only feature, but its full shape can be finalized once the pipeline behaviors (P1–P3) are understood. It is grouped with multilingual as a P3 because the UI feature that consumes it comes later.

**Independent Test**: Using only the exposed programmatic interface, trigger model download/preparation, query readiness/health, start a session, receive streamed segments, fetch the transcript, and stop the session — entirely from a script or test harness.

**Acceptance Scenarios**:

1. **Given** the models are not yet present locally, **When** a consumer invokes the model-preparation operation, **Then** the backend downloads and caches the required speech and diarization models and reports progress and final readiness.
2. **Given** the models are already cached, **When** the backend starts, **Then** it loads them from the local cache without re-downloading and reports a ready state.
3. **Given** a consumer queries system readiness, **When** required models, audio permissions, or compute capabilities are missing, **Then** the response clearly enumerates what is missing and what remains usable.
4. **Given** multiple session-control operations (start, query, stop), **When** they are invoked through the API, **Then** each returns structured results (session identifier, status, segments, errors) suitable for a UI or test to consume.

---

### Edge Cases

- **Silence / no speech**: When no one is speaking, the backend should not fabricate text or spurious speaker labels; it should simply emit nothing (or explicit silence markers) until speech resumes.
- **Single continuous speaker for a long time**: Labels must remain stable; the same voice should not drift into new speaker labels over a long monologue.
- **New speaker appears late**: A voice that first speaks well into the meeting should receive a fresh label and be tracked from that point on.
- **More simultaneous speakers than the diarizer can resolve**: The backend should degrade gracefully (best-effort attribution) and signal lower confidence rather than crash.
- **Unsupported or unexpected language**: When speech is in a language outside the supported set, the backend should produce best-effort output and flag low confidence / unknown language rather than failing the session.
- **Audio device or permission changes mid-session** (mic unplugged, system-audio permission revoked, output device switched): The session should surface a clear error/state change instead of hanging.
- **Model download interrupted** (network loss, partial download): Preparation should fail cleanly and be safely retryable/resumable without leaving a corrupt cache.
- **Insufficient compute / memory** to run the models in real time: The backend should report the constraint and, where possible, fall back to a lower-throughput mode rather than silently dropping audio.
- **Backpressure**: If transcription cannot keep pace with incoming audio, the backend must not lose the audio timeline; it should buffer or signal lag rather than corrupt timestamps or ordering.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST capture live audio from the local microphone on macOS for the duration of a transcription session.
- **FR-002**: System MUST capture system/output audio so that voices from any running meeting application (e.g., Microsoft Teams, Google Meet, Zoom) are included, independent of which specific app is in use.
- **FR-003**: System MUST separate concurrent audio into distinct speakers via speaker diarization, supporting two or more speakers in a single session.
- **FR-004**: System MUST assign each detected speaker a stable, distinguishable label (e.g., "Speaker 1", "Speaker 2", …) that remains consistent for that speaker throughout the session.
- **FR-005**: System MUST transcribe captured speech into text using a multilingual streaming speech-to-text capability, producing results incrementally during the session rather than only after it ends.
- **FR-006**: System MUST attach to each transcript segment a speaker label, a start time, an end time, and the recognized text.
- **FR-007**: System MUST support multilingual transcription across the speech model's supported language set and record an identified language for each segment.
- **FR-008**: System MUST allow a consumer to optionally provide a language hint for a session while still functioning without one (auto language handling).
- **FR-009**: System MUST merge audio from microphone and system sources into a single, time-ordered transcript with consistent speaker labeling across both sources.
- **FR-010**: System MUST expose a programmatic interface to start a session, stream/emit transcript segments as they are produced, retrieve the accumulated transcript, and stop a session.
- **FR-011**: System MUST expose a programmatic operation to ensure the required speech and diarization models are downloaded and cached locally, reporting progress and completion.
- **FR-012**: System MUST load already-cached models without re-downloading on subsequent runs.
- **FR-013**: System MUST expose a readiness/health operation that reports whether required models are available, whether necessary audio permissions are granted, and whether the device can run the models.
- **FR-014**: System MUST run entirely on-device on macOS Apple Silicon, without sending captured audio or transcripts to external services for processing.
- **FR-015**: System MUST handle and report audio-capture permission states for both microphone and system-audio capture, surfacing clear, actionable errors when a required permission is missing.
- **FR-016**: System MUST preserve correct chronological ordering and coherent timestamps of segments even when transcription lags behind real-time input.
- **FR-017**: System MUST return structured, machine-readable results (session identifiers, status, segments, errors) from all exposed operations so a future UI can consume them directly.
- **FR-018**: System MUST detect and report when speech cannot be confidently attributed or recognized (e.g., too many overlapping speakers, unsupported language) via per-segment confidence or status rather than failing silently.
- **FR-019**: System MUST make the live, accumulating transcript available to a consumer at any point during an active session, not only after the session stops.
- **FR-020**: System MUST allow multiple sessions to be managed over the backend's lifetime (sequentially at minimum), each with its own independent transcript and speaker labeling.
- **FR-021**: System MUST detect when available compute/memory is insufficient to sustain real-time inference and respond by degrading gracefully — buffering and/or signaling lag rather than silently dropping audio — and MUST report the constraint to the consumer instead of corrupting the audio timeline.

### Key Entities *(include if feature involves data)*

- **Transcription Session**: A single live capture-and-transcribe run. Has a unique identifier, a status (preparing / active / stopped / error), the configured audio sources (mic, system, or both), an optional language hint, start and stop times, and an ordered collection of transcript segments.
- **Audio Source**: A captured input feeding a session — the local microphone or the system/meeting-app output. Describes which physical/logical source it represents and its current capture state.
- **Speaker**: A distinct voice identified within a session, represented by a stable label (Speaker 1, 2, …). Scoped to the session; not assumed to persist or be recognized across separate sessions.
- **Transcript Segment**: One attributed unit of recognized speech, carrying the speaker label, start time, end time, recognized text, identified language, and a confidence/quality indicator.
- **Model Asset**: A downloadable speech (ASR) or diarization model required by the pipeline. Has an identity/version, a local-cache location, a download/readiness state, and supported-language metadata (for the ASR model).
- **System Readiness Report**: A snapshot of whether the backend can run — model availability, microphone permission, system-audio permission, and compute capability — with a clear list of anything missing.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: During a live session, transcript segments for spoken phrases appear to the consumer within ~2 seconds of the speech finishing (perceived near-real-time).
- **SC-002**: In a meeting with up to 4 distinct speakers, at least 90% of speech time is attributed to the correct, stable speaker label across the session.
- **SC-003**: For clear speech in any supported language, the per-segment language is correctly identified for at least 95% of segments.
- **SC-004**: A consumer can capture both the local speaker and remote meeting-app participants into one merged, time-ordered transcript using only the exposed API, with no manual audio routing beyond granting the documented permissions.
- **SC-005**: First-time model preparation completes and reaches a "ready" state on a single API call, and subsequent backend starts reach "ready" from cache in under 30 seconds without re-downloading.
- **SC-006**: All captured audio and transcripts remain on-device — no meeting audio or transcript content leaves the machine during normal operation (verifiable by the absence of outbound transfer of audio/transcript data).
- **SC-007**: A new developer can drive a full session lifecycle (prepare models → check readiness → start → receive segments → stop) end-to-end through the documented API in under 15 minutes using only the provided interface.
- **SC-008**: The backend sustains a continuous 60-minute session without losing audio-timeline continuity, dropping speaker labels, or corrupting segment ordering.

## Assumptions

- **Scope boundary — backend only**: This feature delivers the audio-capture, diarization, transcription, and API layers only. No graphical app window, "liquid retina" visual, shake/visualization behavior, or other UI is in scope; those described visuals are noted as motivation for the *next* feature and are explicitly deferred.
- **Speech model**: The multilingual streaming STT capability is provided by NVIDIA Nemotron 3.5 ASR Streaming (0.6B), which supports ~40 language-locales via language-ID conditioning, streaming chunked inference, and native punctuation/capitalization. This is the chosen realization of the "multilingual streaming STT" requirement.
- **Diarization approach**: Based on current research, real-time online diarization is provided by NVIDIA Streaming Sortformer (arrival-order speaker cache, designed for streaming, handles overlapping speech and up to ~4 concurrent speakers), which aligns with the NeMo/Nemotron stack. Pyannote 3.1 is the fallback/alternative if Sortformer constraints prove limiting. Final selection is confirmed during planning.
- **Speaker identity is session-scoped and anonymous**: Speakers are labeled Speaker 1, 2, … per session. Persistent voice enrollment or recognizing the same person across separate meetings is out of scope for this feature.
- **System-audio capture method**: Capturing remote meeting-app participants is assumed to use macOS's supported system/screen-audio capture facilities (requiring user permission), rather than per-app integrations with Teams/Meet/Zoom. The backend is meeting-app-agnostic.
- **Target platform**: macOS on Apple Silicon (M1+) is the primary and only required target for this feature; other platforms are out of scope. System-audio capture via Core Audio Process Taps requires **macOS 14.4 or later**; on macOS 13.0–14.3 a ScreenCaptureKit-based fallback is used for system-audio capture. macOS 13 Ventura is the minimum supported version.
- **On-device processing**: All inference runs locally; the only network access is the one-time model download from the model hub during preparation.
- **Downstream summarization out of scope**: Producing meeting minutes / LLM summaries from the transcript is a later concern; this feature stops at the diarized, multilingual transcript and its API.
- **Consumer & API style**: The API's first consumer is the future in-app UI (and automated tests); the interface is designed to be driven programmatically, not by end users directly. The exposed surface is an **in-process Python library** (classes/methods the consumer imports and calls directly), rather than a separate local server or native bridge — chosen for lowest latency and simplest integration with the model runtime. Live segment delivery is provided via in-process streaming (e.g., callbacks/iterators) rather than a network socket.
- **Existing prototype**: The repository's current Whisper-based `transcribe_meeting.py` is a file-based batch prototype and will be superseded by this streaming, diarized, Nemotron-based backend.
