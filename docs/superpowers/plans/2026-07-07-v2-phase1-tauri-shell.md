# ASR-Memo v2 — Phase 1: Workspace + Tauri Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Cargo workspace with `asr-memo-core` (contract DTOs, backend traits, deterministic fakes, session orchestrator) and a Tauri 2.x app that runs the *existing* `app/web` UI unchanged via a pywebview shim, streaming fake diarized segments end-to-end, with a 3-OS CI matrix.

**Architecture:** Rust library crate (`asr-memo-core`) holds everything testable without a window: JSON DTOs locked to `app/dto.py`, the `AudioCapture`/`SpeechTranscriber`/`SpeakerDiarizer` traits, fakes, and a `Session` that fuses tokens+speaker frames into `SegmentDto`s. The Tauri crate (`asr-memo-app`) is a thin adapter: commands mirror `app/bridge.py:Api` return shapes; events go out as one `backend-event` Tauri event whose payload is exactly the dict `window.onBackendEvent` already handles. A `tauri-shim.js` fabricates `window.pywebview.api` so `app.js` runs byte-identical under both shells.

**Tech Stack:** Rust (edition 2021, MSRV 1.77.2 per Tauri 2), Tauri 2.x (`withGlobalTauri`), serde/serde_json, thiserror, std threads + `std::sync::mpsc` (no tokio in Phase 1).

## Global Constraints

- The Python app must keep working: **do not modify or move anything under `app/` or `src/meeting_asr/`** — `app/web` is *copied*, not moved.
- JSON contracts are locked to `app/dto.py` / `app/bridge.py`: `SegmentDTO`, `ReadinessDTO`, `ErrorInfo`, event dicts `{"type": "status"|"segment"|"progress"|"prepare_progress"|"prepare_done"|"error", ...}`, command returns (`start_live → {"app_session_id"}`, `stop_session → {"status": "stopped"}`, `get_transcript → {"segments": [], "speakers": []}`, `pick_* → {"path"}`, `export_transcript → {"path"}` or `{"error"}`).
- Status strings: `"starting"`, `"recording"`, `"stopping"`, `"stopped"`, `"processing"`, `"error"` (match `app/bridge.py:AppStatus` usage in `app/web/app.js`).
- Speaker labels are `"S1"`, `"S2"`, … (arrival order), max 4 — matches the diarizer contract and the UI palette.
- No network, no models, no audio hardware in any test. All timestamps `f64` seconds; all PCM `f32` 16 kHz mono.
- Commit after every task; conventional-commit messages.
- Run Rust commands from the repo root. If `cargo` is missing, install via `rustup` (https://rustup.rs), stable toolchain.

---

### Task 1: Cargo workspace + `asr-memo-core` + contract DTOs

**Files:**
- Create: `Cargo.toml` (repo root, workspace)
- Create: `crates/asr-memo-core/Cargo.toml`
- Create: `crates/asr-memo-core/src/lib.rs`
- Create: `crates/asr-memo-core/src/types.rs`
- Modify: `.gitignore` (add `/target`)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `asr_memo_core::types::{AudioSource, SegmentDto, SpeakerDto, ModelDto, ReadinessDto, ErrorInfoDto, PrepareProgressDto}` — serde-serializable structs whose JSON matches `app/dto.py` exactly. All later tasks import these.

- [ ] **Step 1: Create the workspace and crate skeleton**

Root `Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["crates/asr-memo-core"]

[workspace.package]
edition = "2021"
rust-version = "1.77.2"
license = "MIT"

[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "2.0"
```

`crates/asr-memo-core/Cargo.toml`:

```toml
[package]
name = "asr-memo-core"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
```

`crates/asr-memo-core/src/lib.rs`:

```rust
//! asr-memo-core: platform-independent core for ASR-Memo v2.
//! JSON contracts here are locked to the Python bridge (`app/dto.py`).

pub mod types;
```

Append to `.gitignore`:

```
/target
```

- [ ] **Step 2: Write the failing contract test**

Create `crates/asr-memo-core/src/types.rs` containing ONLY the test for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Golden contract: must match app/dto.py::segment_dto exactly.
    #[test]
    fn segment_dto_matches_python_bridge_contract() {
        let seg = SegmentDto {
            segment_id: "seg-1".into(),
            speaker_label: "S1".into(),
            start: 0.0,
            end: 1.25,
            text: "hello".into(),
            language: Some("en".into()),
            confidence: 0.9,
            confidence_band: Some("high".into()),
            source: Some(AudioSource::System),
            is_final: true,
        };
        let v = serde_json::to_value(&seg).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "segment_id": "seg-1", "speaker_label": "S1",
                "start": 0.0, "end": 1.25, "text": "hello",
                "language": "en", "confidence": 0.9,
                "confidence_band": "high", "source": "system",
                "is_final": true
            })
        );
    }

    /// Golden contract: must match app/dto.py::readiness_dto exactly.
    #[test]
    fn readiness_dto_matches_python_bridge_contract() {
        let r = ReadinessDto {
            ready: false,
            compute_backend: "fake".into(),
            os_supports_process_tap: true,
            mic_permission: true,
            system_audio_permission: false,
            models: vec![ModelDto {
                name: "asr".into(),
                kind: "asr".into(),
                state: "missing".into(),
                is_cached: false,
            }],
            missing: vec!["system audio permission".into()],
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "ready": false, "compute_backend": "fake",
                "os_supports_process_tap": true, "mic_permission": true,
                "system_audio_permission": false,
                "models": [{"name": "asr", "kind": "asr", "state": "missing", "is_cached": false}],
                "missing": ["system audio permission"]
            })
        );
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p asr-memo-core`
Expected: FAIL to compile — `cannot find type SegmentDto` (compile error is the failing state for a new module).

- [ ] **Step 4: Write the DTO types (top of `types.rs`, above the tests)**

```rust
use serde::{Deserialize, Serialize};

/// Audio origin. Serializes as "microphone" / "system" (app/dto.py contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioSource {
    Microphone,
    System,
}

/// SegmentDTO — the atomic rendered/exported unit (app/dto.py::segment_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentDto {
    pub segment_id: String,
    pub speaker_label: String,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub language: Option<String>,
    pub confidence: f64,
    pub confidence_band: Option<String>,
    pub source: Option<AudioSource>,
    pub is_final: bool,
}

/// SpeakerDTO (app/dto.py::speaker_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerDto {
    pub label: String,
    pub color: String,
    pub total_speech_seconds: f64,
    pub segment_count: u32,
}

/// One model row inside ReadinessDTO (app/dto.py::_model_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDto {
    pub name: String,
    pub kind: String,
    pub state: String,
    pub is_cached: bool,
}

/// ReadinessDTO — drives the setup screen (app/dto.py::readiness_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadinessDto {
    pub ready: bool,
    pub compute_backend: String,
    pub os_supports_process_tap: bool,
    pub mic_permission: bool,
    pub system_audio_permission: bool,
    pub models: Vec<ModelDto>,
    pub missing: Vec<String>,
}

/// ErrorInfo (app/dto.py::error_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorInfoDto {
    pub code: String,
    pub message: String,
    pub recoverable: bool,
    pub hint: Option<String>,
}

/// prepare_progress event payload (app/dto.py::prepare_progress_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepareProgressDto {
    pub asset: String,
    pub downloaded: u64,
    pub total: u64,
    pub fraction: f64,
    pub state: String,
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p asr-memo-core`
Expected: `test result: ok. 2 passed`

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock .gitignore crates/
git commit -m "feat(v2): cargo workspace + asr-memo-core with contract-locked DTOs"
```

---

### Task 2: Backend traits + deterministic fakes

**Files:**
- Create: `crates/asr-memo-core/src/traits.rs`
- Create: `crates/asr-memo-core/src/fakes.rs`
- Modify: `crates/asr-memo-core/src/lib.rs` (add `pub mod traits; pub mod fakes;`)

**Interfaces:**
- Consumes: `types::AudioSource` from Task 1.
- Produces (used by Tasks 3 and 5):
  - `traits::AudioFrame { pcm: Vec<f32>, t_start: f64, t_end: f64, source: AudioSource }`
  - `traits::AsrToken { text: String, t_start: f64, t_end: f64, confidence: f64 }`
  - `traits::DiarFrame { t: f64, speaker: Option<u8> }`
  - `traits::CoreError` (thiserror enum, variant `Capture { code, message }`)
  - `trait SpeechTranscriber: Send { fn reset(&mut self); fn push(&mut self, frame: &AudioFrame) -> Vec<AsrToken>; fn flush(&mut self) -> Vec<AsrToken>; }`
  - `trait SpeakerDiarizer: Send { fn reset(&mut self); fn push(&mut self, frame: &AudioFrame) -> Vec<DiarFrame>; }`
  - `trait AudioCapture: Send { fn start(&mut self, on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError>; fn stop(&mut self); }`
  - `fakes::FakeCapture::from_script(Vec<AudioFrame>)`, `fakes::FakeTranscriber::with_tokens(Vec<AsrToken>)`, `fakes::FakeDiarizer::alternating(period_secs: f64)`
  - `fakes::demo_script(seconds: f64) -> (Vec<AudioFrame>, Vec<AsrToken>)` — the canned live-demo content Task 5 streams to the UI.

- [ ] **Step 1: Write the failing tests**

Create `crates/asr-memo-core/src/fakes.rs` with tests only:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::*;
    use crate::types::AudioSource;
    use std::sync::mpsc;

    fn frame(t0: f64, t1: f64) -> AudioFrame {
        let n = ((t1 - t0) * 16000.0) as usize;
        AudioFrame { pcm: vec![0.0; n], t_start: t0, t_end: t1, source: AudioSource::Microphone }
    }

    #[test]
    fn fake_capture_emits_script_then_stops() {
        let script = vec![frame(0.0, 0.5), frame(0.5, 1.0)];
        let mut cap = FakeCapture::from_script(script);
        let (tx, rx) = mpsc::channel();
        cap.start(Box::new(move |f| tx.send(f).unwrap())).unwrap();
        cap.stop(); // joins the emit thread
        let got: Vec<AudioFrame> = rx.try_iter().collect();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].t_start, 0.0);
        assert_eq!(got[1].t_end, 1.0);
    }

    #[test]
    fn fake_transcriber_releases_tokens_by_frame_time() {
        let tokens = vec![
            AsrToken { text: "hello".into(), t_start: 0.2, t_end: 0.5, confidence: 0.9 },
            AsrToken { text: "world".into(), t_start: 0.6, t_end: 0.9, confidence: 0.8 },
        ];
        let mut asr = FakeTranscriber::with_tokens(tokens);
        let out1 = asr.push(&frame(0.0, 0.5));
        assert_eq!(out1.len(), 1); // only "hello" has t_end <= 0.5
        assert_eq!(out1[0].text, "hello");
        let out2 = asr.flush();
        assert_eq!(out2.len(), 1);
        assert_eq!(out2[0].text, "world");
    }

    #[test]
    fn fake_diarizer_alternates_speakers_by_period() {
        let mut diar = FakeDiarizer::alternating(1.0);
        let frames = diar.push(&frame(0.0, 2.0));
        // one DiarFrame per 80 ms over [0.0, 2.0) → 25 frames
        assert_eq!(frames.len(), 25);
        assert_eq!(frames[0].speaker, Some(0)); // t=0.00 → first period
        assert_eq!(frames[13].speaker, Some(1)); // t=1.04 → second period
    }

    #[test]
    fn demo_script_is_nonempty_and_time_ordered() {
        let (frames, tokens) = demo_script(4.0);
        assert!(!frames.is_empty() && !tokens.is_empty());
        assert!(frames.windows(2).all(|w| w[0].t_end <= w[1].t_start + 1e-9));
        assert!(tokens.windows(2).all(|w| w[0].t_start <= w[1].t_start));
    }
}
```

Add to `lib.rs`: `pub mod traits;` and `pub mod fakes;` (create an empty `traits.rs` so it compiles far enough to show the real failure).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p asr-memo-core`
Expected: FAIL to compile — `cannot find struct FakeCapture` / missing trait items.

- [ ] **Step 3: Implement `traits.rs`**

```rust
//! Backend traits — the Rust equivalents of src/meeting_asr's protocols.
//! Real implementations arrive in Phases 2–4; Phase 1 ships fakes only.

use crate::types::AudioSource;

/// A block of 16 kHz mono f32 PCM on the session clock.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFrame {
    pub pcm: Vec<f32>,
    pub t_start: f64,
    pub t_end: f64,
    pub source: AudioSource,
}

/// One decoded word with timestamps (the aligner's input unit).
#[derive(Debug, Clone, PartialEq)]
pub struct AsrToken {
    pub text: String,
    pub t_start: f64,
    pub t_end: f64,
    pub confidence: f64,
}

/// One 80 ms diarizer frame: active speaker index (arrival order) or silence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiarFrame {
    pub t: f64,
    pub speaker: Option<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("{code}: {message}")]
    Capture { code: String, message: String },
}

pub trait SpeechTranscriber: Send {
    fn reset(&mut self);
    /// Push one frame; returns the tokens finalized by this push.
    fn push(&mut self, frame: &AudioFrame) -> Vec<AsrToken>;
    /// End of stream; returns any remaining tokens.
    fn flush(&mut self) -> Vec<AsrToken>;
}

pub trait SpeakerDiarizer: Send {
    fn reset(&mut self);
    /// Push one frame; returns the 80 ms speaker frames it covers.
    fn push(&mut self, frame: &AudioFrame) -> Vec<DiarFrame>;
}

pub trait AudioCapture: Send {
    /// Start delivering frames to `on_frame` (from any thread). Idempotent stop.
    fn start(&mut self, on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError>;
    fn stop(&mut self);
}
```

- [ ] **Step 4: Implement `fakes.rs` (above the tests from Step 1)**

```rust
//! Deterministic fakes mirroring tests/_fakes.py — no audio, no models, no sleep.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::traits::*;
use crate::types::AudioSource;

/// Emits a scripted frame sequence on a background thread, then finishes.
pub struct FakeCapture {
    script: Option<Vec<AudioFrame>>,
    handle: Option<JoinHandle<()>>,
    stopped: Arc<AtomicBool>,
}

impl FakeCapture {
    pub fn from_script(script: Vec<AudioFrame>) -> Self {
        Self { script: Some(script), handle: None, stopped: Arc::new(AtomicBool::new(false)) }
    }
}

impl AudioCapture for FakeCapture {
    fn start(&mut self, mut on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError> {
        let script = self.script.take().ok_or_else(|| CoreError::Capture {
            code: "capture.restart".into(),
            message: "FakeCapture can only start once".into(),
        })?;
        let stopped = self.stopped.clone();
        self.handle = Some(std::thread::spawn(move || {
            for f in script {
                if stopped.load(Ordering::Acquire) {
                    break;
                }
                on_frame(f);
            }
        }));
        Ok(())
    }

    fn stop(&mut self) {
        self.stopped.store(true, Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Releases scripted tokens once the stream time passes their t_end.
pub struct FakeTranscriber {
    pending: Vec<AsrToken>,
}

impl FakeTranscriber {
    pub fn with_tokens(mut tokens: Vec<AsrToken>) -> Self {
        tokens.sort_by(|a, b| a.t_end.total_cmp(&b.t_end));
        Self { pending: tokens }
    }
}

impl SpeechTranscriber for FakeTranscriber {
    fn reset(&mut self) {
        self.pending.clear();
    }

    fn push(&mut self, frame: &AudioFrame) -> Vec<AsrToken> {
        let split = self.pending.partition_point(|t| t.t_end <= frame.t_end);
        self.pending.drain(..split).collect()
    }

    fn flush(&mut self) -> Vec<AsrToken> {
        std::mem::take(&mut self.pending)
    }
}

/// Speaker 0/1 alternating every `period_secs`; one DiarFrame per 80 ms.
pub struct FakeDiarizer {
    period_secs: f64,
}

impl FakeDiarizer {
    pub fn alternating(period_secs: f64) -> Self {
        Self { period_secs }
    }
}

impl SpeakerDiarizer for FakeDiarizer {
    fn reset(&mut self) {}

    fn push(&mut self, frame: &AudioFrame) -> Vec<DiarFrame> {
        const STEP: f64 = 0.08;
        let mut out = Vec::new();
        let mut t = frame.t_start;
        while t < frame.t_end - 1e-9 {
            let speaker = ((t / self.period_secs) as u64 % 2) as u8;
            out.push(DiarFrame { t, speaker: Some(speaker) });
            t += STEP;
        }
        out
    }
}

/// Canned live-demo content: 0.5 s silent frames plus a scripted two-speaker
/// conversation, `seconds` long. Task 5 streams this to the UI in fake mode.
pub fn demo_script(seconds: f64) -> (Vec<AudioFrame>, Vec<AsrToken>) {
    let mut frames = Vec::new();
    let mut t = 0.0;
    while t < seconds {
        let t1 = (t + 0.5).min(seconds);
        frames.push(AudioFrame {
            pcm: vec![0.0; ((t1 - t) * 16000.0) as usize],
            t_start: t,
            t_end: t1,
            source: AudioSource::Microphone,
        });
        t = t1;
    }
    let words = ["this", "is", "a", "fake", "session", "streaming", "live", "segments"];
    let tokens = words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let t0 = 0.2 + i as f64 * (seconds - 0.4) / words.len() as f64;
            AsrToken { text: (*w).into(), t_start: t0, t_end: t0 + 0.25, confidence: 0.92 }
        })
        .filter(|tok| tok.t_end < seconds)
        .collect();
    (frames, tokens)
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p asr-memo-core`
Expected: `test result: ok.` (6 tests: 2 from Task 1 + 4 new)

- [ ] **Step 6: Commit**

```bash
git add crates/asr-memo-core
git commit -m "feat(v2): backend traits + deterministic fakes (capture/asr/diarizer)"
```

---

### Task 3: Session orchestrator with midpoint aligner

**Files:**
- Create: `crates/asr-memo-core/src/session.rs`
- Modify: `crates/asr-memo-core/src/lib.rs` (add `pub mod session;`)

**Interfaces:**
- Consumes: Task 1 DTOs; Task 2 traits + fakes.
- Produces (used by Task 5):
  - `session::SessionEvent` enum: `Status(String)` | `Segment(SegmentDto)` | `Error(ErrorInfoDto)`
  - `session::Session::run(capture: Box<dyn AudioCapture>, transcriber: Box<dyn SpeechTranscriber>, diarizer: Box<dyn SpeakerDiarizer>, language_hint: Option<String>, on_event: Box<dyn FnMut(SessionEvent) + Send>) -> Result<Session, CoreError>`
  - `Session::stop(self)` — stops capture, flushes, emits final segments + `Status("stopped")`.

> The midpoint aligner below is Phase 1's real, working fusion: each token gets
> the speaker active at its midpoint; consecutive same-speaker tokens merge into
> one segment, split on speaker change or a >1.0 s gap. Phase 4 replaces it with
> the full `fusion/aligner.py` port behind the same `SessionEvent` interface.

- [ ] **Step 1: Write the failing test (bottom of new `session.rs`)**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fakes::*;
    use crate::traits::*;
    use crate::types::AudioSource;
    use std::sync::mpsc;

    #[test]
    fn session_fuses_tokens_and_speakers_into_segments() {
        let frames: Vec<AudioFrame> = (0..4)
            .map(|i| AudioFrame {
                pcm: vec![0.0; 8000],
                t_start: i as f64 * 0.5,
                t_end: (i + 1) as f64 * 0.5,
                source: AudioSource::Microphone,
            })
            .collect();
        let tokens = vec![
            AsrToken { text: "hello".into(), t_start: 0.10, t_end: 0.40, confidence: 0.9 },
            AsrToken { text: "there".into(), t_start: 0.45, t_end: 0.80, confidence: 0.9 },
            AsrToken { text: "hi".into(), t_start: 1.20, t_end: 1.50, confidence: 0.7 },
        ];
        let (tx, rx) = mpsc::channel();
        let session = Session::run(
            Box::new(FakeCapture::from_script(frames)),
            Box::new(FakeTranscriber::with_tokens(tokens)),
            Box::new(FakeDiarizer::alternating(1.0)), // speaker flips at t=1.0
            Some("en".into()),
            Box::new(move |e| tx.send(e).unwrap()),
        )
        .unwrap();
        session.stop();

        let events: Vec<SessionEvent> = rx.try_iter().collect();
        let segments: Vec<&SegmentDto> = events
            .iter()
            .filter_map(|e| match e {
                SessionEvent::Segment(s) => Some(s),
                _ => None,
            })
            .collect();
        assert_eq!(segments.len(), 2, "speaker change at t=1.0 splits segments");
        assert_eq!(segments[0].text, "hello there");
        assert_eq!(segments[0].speaker_label, "S1");
        assert_eq!(segments[1].text, "hi");
        assert_eq!(segments[1].speaker_label, "S2");
        assert!(segments[0].is_final && segments[1].is_final);
        assert_eq!(segments[0].language.as_deref(), Some("en"));
        // Final status is "stopped"
        assert!(matches!(events.last(), Some(SessionEvent::Status(s)) if s == "stopped"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p asr-memo-core session`
Expected: FAIL to compile — `Session` / `SessionEvent` not found.

- [ ] **Step 3: Implement the session (top of `session.rs`)**

```rust
//! Session orchestrator: capture → (transcriber ∥ diarizer) → fused SegmentDto
//! events. Phase-1 fusion is the midpoint aligner (see plan note); the event
//! interface is what Phase 4's full aligner port will slot into.

use std::sync::mpsc;
use std::thread::JoinHandle;

use crate::traits::*;
use crate::types::{ErrorInfoDto, SegmentDto};

#[derive(Debug)]
pub enum SessionEvent {
    Status(String),
    Segment(SegmentDto),
    Error(ErrorInfoDto),
}

const SEGMENT_GAP_SECS: f64 = 1.0;

/// Groups tokens by the speaker active at each token's midpoint.
struct MidpointAligner {
    diar: Vec<DiarFrame>,
    pending: Vec<(AsrToken, String)>, // token + speaker label
    seen_speakers: Vec<u8>,           // arrival order → "S1".."S4"
    next_segment_id: u64,
    language: Option<String>,
}

impl MidpointAligner {
    fn new(language: Option<String>) -> Self {
        Self { diar: Vec::new(), pending: Vec::new(), seen_speakers: Vec::new(), next_segment_id: 1, language }
    }

    fn label_for(&mut self, raw: u8) -> String {
        if !self.seen_speakers.contains(&raw) {
            self.seen_speakers.push(raw);
        }
        let idx = self.seen_speakers.iter().position(|s| *s == raw).unwrap();
        format!("S{}", idx + 1)
    }

    fn speaker_at(&self, t: f64) -> Option<u8> {
        self.diar.iter().rev().find(|f| f.t <= t).and_then(|f| f.speaker)
    }

    fn feed(&mut self, tokens: Vec<AsrToken>, diar: Vec<DiarFrame>) -> Vec<SegmentDto> {
        self.diar.extend(diar);
        let mut out = Vec::new();
        for tok in tokens {
            let mid = (tok.t_start + tok.t_end) / 2.0;
            let label = match self.speaker_at(mid) {
                Some(raw) => self.label_for(raw),
                None => "S1".to_string(), // no diar info yet → first speaker
            };
            let split = match self.pending.last() {
                Some((prev, prev_label)) => {
                    *prev_label != label || tok.t_start - prev.t_end > SEGMENT_GAP_SECS
                }
                None => false,
            };
            if split {
                out.push(self.close_segment());
            }
            self.pending.push((tok, label));
        }
        out
    }

    fn close_segment(&mut self) -> SegmentDto {
        let toks = std::mem::take(&mut self.pending);
        let label = toks[0].1.clone();
        let text = toks.iter().map(|(t, _)| t.text.as_str()).collect::<Vec<_>>().join(" ");
        let confidence =
            toks.iter().map(|(t, _)| t.confidence).sum::<f64>() / toks.len() as f64;
        let band = if confidence >= 0.8 { "high" } else if confidence >= 0.5 { "medium" } else { "low" };
        let seg = SegmentDto {
            segment_id: format!("seg-{}", self.next_segment_id),
            speaker_label: label,
            start: toks[0].0.t_start,
            end: toks.last().unwrap().0.t_end,
            text,
            language: self.language.clone(),
            confidence,
            confidence_band: Some(band.to_string()),
            source: None,
            is_final: true,
        };
        self.next_segment_id += 1;
        seg
    }

    fn flush(&mut self) -> Vec<SegmentDto> {
        if self.pending.is_empty() { Vec::new() } else { vec![self.close_segment()] }
    }
}

pub struct Session {
    capture: Box<dyn AudioCapture>,
    frame_tx: mpsc::Sender<Option<AudioFrame>>,
    worker: JoinHandle<()>,
}

impl Session {
    pub fn run(
        mut capture: Box<dyn AudioCapture>,
        mut transcriber: Box<dyn SpeechTranscriber>,
        mut diarizer: Box<dyn SpeakerDiarizer>,
        language_hint: Option<String>,
        mut on_event: Box<dyn FnMut(SessionEvent) + Send>,
    ) -> Result<Session, CoreError> {
        let (frame_tx, frame_rx) = mpsc::channel::<Option<AudioFrame>>();
        let worker = std::thread::spawn(move || {
            let mut aligner = MidpointAligner::new(language_hint);
            on_event(SessionEvent::Status("recording".into()));
            while let Ok(Some(frame)) = frame_rx.recv() {
                let tokens = transcriber.push(&frame);
                let diar = diarizer.push(&frame);
                for seg in aligner.feed(tokens, diar) {
                    on_event(SessionEvent::Segment(seg));
                }
            }
            // end of stream: flush the transcriber, then the aligner
            for seg in aligner.feed(transcriber.flush(), Vec::new()) {
                on_event(SessionEvent::Segment(seg));
            }
            for seg in aligner.flush() {
                on_event(SessionEvent::Segment(seg));
            }
            on_event(SessionEvent::Status("stopped".into()));
        });

        let cb_tx = frame_tx.clone();
        capture.start(Box::new(move |f| {
            let _ = cb_tx.send(Some(f));
        }))?;

        Ok(Session { capture, frame_tx, worker })
    }

    pub fn stop(mut self) {
        self.capture.stop(); // joins the fake's emit thread → all frames sent
        let _ = self.frame_tx.send(None); // end-of-stream sentinel
        let _ = self.worker.join();
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p asr-memo-core`
Expected: `test result: ok.` (7 tests)

- [ ] **Step 5: Commit**

```bash
git add crates/asr-memo-core
git commit -m "feat(v2): session orchestrator with midpoint fusion over fakes"
```

---

### Task 4: Tauri app crate hosting the existing UI

**Files:**
- Create: `crates/asr-memo-app/Cargo.toml`
- Create: `crates/asr-memo-app/build.rs`
- Create: `crates/asr-memo-app/tauri.conf.json`
- Create: `crates/asr-memo-app/src/main.rs`
- Create: `crates/asr-memo-app/src/lib.rs`
- Create: `crates/asr-memo-app/ui/` (copy of `app/web/` + `tauri-shim.js`; `index.html` gains one `<script>` tag)
- Create: `crates/asr-memo-app/icons/icon.png` (any 512×512 placeholder PNG; `tauri icon` regenerates real ones later)
- Modify: root `Cargo.toml` (add member)

**Interfaces:**
- Consumes: nothing from core yet (window-only task).
- Produces: `asr_memo_app::run()` — launches the Tauri window serving `ui/`; `ui/tauri-shim.js` (populated in Task 5; created as an empty file here so `index.html` doesn't 404).

- [ ] **Step 1: Copy the UI and add the shim script tag**

```bash
mkdir -p crates/asr-memo-app/ui
cp -R app/web/. crates/asr-memo-app/ui/
touch crates/asr-memo-app/ui/tauri-shim.js
```

Edit `crates/asr-memo-app/ui/index.html`: immediately BEFORE the existing `<script src="app.js">` line, add:

```html
<script src="tauri-shim.js"></script>
```

- [ ] **Step 2: Create the crate**

Root `Cargo.toml` members become:

```toml
members = ["crates/asr-memo-core", "crates/asr-memo-app"]
```

`crates/asr-memo-app/Cargo.toml`:

```toml
[package]
name = "asr-memo-app"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
asr-memo-core = { path = "../asr-memo-core" }
tauri = { version = "2", features = [] }
tauri-plugin-dialog = "2"
serde = { workspace = true }
serde_json = { workspace = true }

[lib]
name = "asr_memo_app"
crate-type = ["staticlib", "cdylib", "rlib"]
```

`crates/asr-memo-app/build.rs`:

```rust
fn main() {
    tauri_build::build()
}
```

`crates/asr-memo-app/tauri.conf.json`:

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "ASR-Memo",
  "version": "0.1.0",
  "identifier": "com.asrmemo.app",
  "build": {
    "frontendDist": "./ui"
  },
  "app": {
    "withGlobalTauri": true,
    "windows": [
      {
        "title": "ASR-Memo",
        "width": 1100,
        "height": 760,
        "resizable": true
      }
    ],
    "security": {
      "csp": null
    }
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": ["icons/icon.png"]
  }
}
```

`crates/asr-memo-app/src/main.rs`:

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    asr_memo_app::run()
}
```

`crates/asr-memo-app/src/lib.rs`:

```rust
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("error while running ASR-Memo");
}
```

Placeholder icon (any square PNG works; simplest, from repo root):

```bash
python3 -c "
import zlib, struct
def chunk(t, d):
    c = t + d
    return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c))
w = h = 512
raw = b''.join(b'\x00' + b'\x1e\x66\xcc\xff' * w for _ in range(h))
png = (b'\x89PNG\r\n\x1a\n'
       + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
       + chunk(b'IDAT', zlib.compress(raw))
       + chunk(b'IEND', b''))
open('crates/asr-memo-app/icons/icon.png', 'wb').write(png)
" && mkdir -p crates/asr-memo-app/icons 2>/dev/null; ls -l crates/asr-memo-app/icons/icon.png
```

(Run `mkdir -p crates/asr-memo-app/icons` first if the one-liner complains about the directory.)

- [ ] **Step 3: Verify it builds and the window opens**

Run: `cargo build -p asr-memo-app`
Expected: compiles clean (first Tauri build takes minutes).
On Linux install Tauri system deps first: `sudo apt-get install -y libwebkit2gtk-4.1-dev build-essential libssl-dev libayatana-appindicator3-dev librsvg2-dev`.

Manual check: `cargo run -p asr-memo-app`
Expected: a window titled "ASR-Memo" showing the existing UI's setup screen (it will sit at "checking readiness" — the API shim is empty until Task 5; that's correct for this task).

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock crates/asr-memo-app
git commit -m "feat(v2): tauri app crate hosting the existing web UI"
```

---

### Task 5: Commands + event bridge (fake mode, end-to-end)

**Files:**
- Create: `crates/asr-memo-app/src/state.rs`
- Create: `crates/asr-memo-app/src/commands.rs`
- Modify: `crates/asr-memo-app/src/lib.rs`
- Modify: `crates/asr-memo-app/ui/tauri-shim.js` (fill in)
- Test: `crates/asr-memo-app/src/commands.rs` (unit tests on the pure logic)

**Interfaces:**
- Consumes: Task 1 DTOs, Task 2 fakes (`demo_script`, `FakeCapture`, `FakeTranscriber`, `FakeDiarizer`), Task 3 `Session`/`SessionEvent`.
- Produces: Tauri commands `get_readiness`, `prepare`, `start_live`, `stop_session`, `get_transcript`, `export_transcript`, `transcribe_file`, `pick_audio_file`, `pick_export_path` (all `rename_all = "snake_case"`); Tauri event `"backend-event"` with payloads identical to `app/bridge.py:_emit` dicts.

- [ ] **Step 1: Write the failing unit tests (bottom of new `commands.rs`)**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_readiness_is_ready() {
        let r = fake_readiness();
        assert!(r.ready);
        assert_eq!(r.compute_backend, "fake");
        assert!(r.missing.is_empty());
    }

    #[test]
    fn segment_event_payload_matches_bridge_contract() {
        let seg = asr_memo_core::types::SegmentDto {
            segment_id: "seg-1".into(),
            speaker_label: "S1".into(),
            start: 0.0,
            end: 1.0,
            text: "hi".into(),
            language: None,
            confidence: 0.9,
            confidence_band: Some("high".into()),
            source: None,
            is_final: true,
        };
        let v = event_payload(&asr_memo_core::session::SessionEvent::Segment(seg));
        assert_eq!(v["type"], "segment");
        assert_eq!(v["segment"]["speaker_label"], "S1");
    }

    #[test]
    fn status_and_error_event_payloads_match_bridge_contract() {
        let v = event_payload(&asr_memo_core::session::SessionEvent::Status("recording".into()));
        assert_eq!(v, serde_json::json!({"type": "status", "status": "recording"}));
        let e = asr_memo_core::types::ErrorInfoDto {
            code: "x".into(), message: "m".into(), recoverable: true, hint: None,
        };
        let v = event_payload(&asr_memo_core::session::SessionEvent::Error(e));
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["code"], "x");
    }

    #[test]
    fn export_markdown_writes_speaker_lines() {
        let seg = asr_memo_core::types::SegmentDto {
            segment_id: "seg-1".into(),
            speaker_label: "S1".into(),
            start: 0.0,
            end: 1.5,
            text: "hello world".into(),
            language: Some("en".into()),
            confidence: 0.9,
            confidence_band: Some("high".into()),
            source: None,
            is_final: true,
        };
        let md = render_markdown(&[seg]);
        assert!(md.contains("**S1**"));
        assert!(md.contains("hello world"));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p asr-memo-app`
Expected: FAIL to compile — `fake_readiness` / `event_payload` / `render_markdown` not found.

- [ ] **Step 3: Implement `state.rs`**

```rust
//! Shared app state: the running session and the accumulated transcript.

use std::sync::Mutex;

use asr_memo_core::session::Session;
use asr_memo_core::types::SegmentDto;

#[derive(Default)]
pub struct AppState {
    pub session: Mutex<Option<Session>>,
    pub transcript: Mutex<Vec<SegmentDto>>,
}
```

- [ ] **Step 4: Implement `commands.rs` (above the tests)**

```rust
//! Tauri commands mirroring app/bridge.py:Api — same names, args, return
//! shapes, and events, so app/web/app.js runs unchanged. Phase 1 wires the
//! deterministic fakes; Phases 2–4 swap in real backends behind these commands.

use tauri::{AppHandle, Emitter, Manager, Runtime, State};

use asr_memo_core::fakes::{demo_script, FakeCapture, FakeDiarizer, FakeTranscriber};
use asr_memo_core::session::{Session, SessionEvent};
use asr_memo_core::types::{ErrorInfoDto, ReadinessDto, SegmentDto, SpeakerDto};

use crate::state::AppState;

// ---------- pure logic (unit-tested) ----------

pub fn fake_readiness() -> ReadinessDto {
    ReadinessDto {
        ready: true,
        compute_backend: "fake".into(),
        os_supports_process_tap: cfg!(target_os = "macos"),
        mic_permission: true,
        system_audio_permission: true,
        models: vec![],
        missing: vec![],
    }
}

/// Serialize a SessionEvent as the exact dict app/bridge.py:_emit sends.
pub fn event_payload(event: &SessionEvent) -> serde_json::Value {
    match event {
        SessionEvent::Status(s) => serde_json::json!({"type": "status", "status": s}),
        SessionEvent::Segment(seg) => serde_json::json!({"type": "segment", "segment": seg}),
        SessionEvent::Error(e) => serde_json::json!({"type": "error", "error": e}),
    }
}

pub fn render_markdown(segments: &[SegmentDto]) -> String {
    let mut out = String::from("# Transcript\n\n");
    for s in segments {
        out.push_str(&format!(
            "- [{:.1}s–{:.1}s] **{}**: {}\n",
            s.start, s.end, s.speaker_label, s.text
        ));
    }
    out
}

fn speakers_view(segments: &[SegmentDto]) -> Vec<SpeakerDto> {
    // Canonical palette from src/meeting_asr/export/palette.py::SPEAKER_COLORS.
    const COLORS: [&str; 4] = ["#1A7F64", "#2D7FF9", "#B4515C", "#9B59B6"];
    let mut labels: Vec<String> = Vec::new();
    for s in segments {
        if !labels.contains(&s.speaker_label) {
            labels.push(s.speaker_label.clone());
        }
    }
    labels
        .into_iter()
        .enumerate()
        .map(|(i, label)| {
            let segs: Vec<&SegmentDto> =
                segments.iter().filter(|s| s.speaker_label == label).collect();
            SpeakerDto {
                color: COLORS[i % COLORS.len()].into(),
                total_speech_seconds: segs.iter().map(|s| s.end - s.start).sum(),
                segment_count: segs.len() as u32,
                label,
            }
        })
        .collect()
}

fn err(code: &str, message: &str, hint: &str) -> serde_json::Value {
    serde_json::json!({"error": ErrorInfoDto {
        code: code.into(), message: message.into(), recoverable: true, hint: Some(hint.into()),
    }})
}

// ---------- Tauri commands ----------

#[tauri::command(rename_all = "snake_case")]
pub fn get_readiness() -> ReadinessDto {
    fake_readiness()
}

#[tauri::command(rename_all = "snake_case")]
pub fn prepare<R: Runtime>(app: AppHandle<R>) -> serde_json::Value {
    // Fake mode: everything is already "ready"; complete immediately.
    let _ = app.emit(
        "backend-event",
        serde_json::json!({"type": "prepare_done", "readiness": fake_readiness()}),
    );
    serde_json::json!({"status": "prepared"})
}

#[tauri::command(rename_all = "snake_case")]
pub fn start_live<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    sources: Vec<String>,
    language_hint: Option<String>,
) -> serde_json::Value {
    let _ = sources; // fake mode ignores source selection
    if state.session.lock().unwrap().is_some() {
        return err("busy", "a session is already running", "Stop the current session first.");
    }
    state.transcript.lock().unwrap().clear();

    let (frames, tokens) = demo_script(8.0);
    let app_for_events = app.clone();
    let session = Session::run(
        Box::new(FakeCapture::from_script(frames)),
        Box::new(FakeTranscriber::with_tokens(tokens)),
        Box::new(FakeDiarizer::alternating(2.0)),
        language_hint,
        Box::new(move |event| {
            if let SessionEvent::Segment(seg) = &event {
                let st: State<'_, AppState> = app_for_events.state();
                st.transcript.lock().unwrap().push(seg.clone());
            }
            let _ = app_for_events.emit("backend-event", event_payload(&event));
        }),
    );
    match session {
        Ok(s) => {
            *state.session.lock().unwrap() = Some(s);
            serde_json::json!({"app_session_id": "live-1"})
        }
        Err(e) => err("start.failed", &e.to_string(), "Try again."),
    }
}

#[tauri::command(rename_all = "snake_case")]
pub fn stop_session(state: State<'_, AppState>) -> serde_json::Value {
    if let Some(s) = state.session.lock().unwrap().take() {
        s.stop(); // emits final segments + status=stopped via on_event
    }
    serde_json::json!({"status": "stopped"})
}

#[tauri::command(rename_all = "snake_case")]
pub fn get_transcript(state: State<'_, AppState>) -> serde_json::Value {
    let segments = state.transcript.lock().unwrap().clone();
    serde_json::json!({"speakers": speakers_view(&segments), "segments": segments})
}

#[tauri::command(rename_all = "snake_case")]
pub fn export_transcript(
    state: State<'_, AppState>,
    path: String,
    format: Option<String>,
) -> serde_json::Value {
    let segments = state.transcript.lock().unwrap().clone();
    if segments.is_empty() {
        return err("export.empty", "no transcript to export", "Produce a transcript first, then export.");
    }
    let fmt = format.unwrap_or_else(|| {
        if path.ends_with(".json") { "json".into() } else { "markdown".into() }
    });
    let body = match fmt.as_str() {
        "json" => serde_json::to_string_pretty(
            &serde_json::json!({"speakers": speakers_view(&segments), "segments": segments}),
        )
        .unwrap(),
        "markdown" | "md" => render_markdown(&segments),
        other => {
            return err("export.format", &format!("unknown format: {other}"), "Choose Markdown (.md) or JSON (.json).")
        }
    };
    match std::fs::write(&path, body) {
        Ok(()) => serde_json::json!({"path": path}),
        Err(e) => err("export.write_failed", &e.to_string(), "Could not write the file — check disk space and the chosen path."),
    }
}

#[tauri::command(rename_all = "snake_case")]
pub fn transcribe_file(path: String, language_hint: Option<String>) -> serde_json::Value {
    let _ = (path, language_hint);
    err("not_implemented", "file import lands in Phase 3 (ASR port)", "Use Start (live fake demo) in Phase 1.")
}

#[tauri::command(rename_all = "snake_case")]
pub fn pick_audio_file<R: Runtime>(app: AppHandle<R>) -> serde_json::Value {
    use tauri_plugin_dialog::DialogExt;
    let path = app
        .dialog()
        .file()
        .add_filter("Audio", &["wav", "flac", "mp3", "m4a"])
        .blocking_pick_file()
        .and_then(|p| p.as_path().map(|p| p.to_string_lossy().into_owned()));
    serde_json::json!({"path": path})
}

#[tauri::command(rename_all = "snake_case")]
pub fn pick_export_path<R: Runtime>(app: AppHandle<R>, format: String) -> serde_json::Value {
    use tauri_plugin_dialog::DialogExt;
    let ext = if format == "json" { "json" } else { "md" };
    let path = app
        .dialog()
        .file()
        .add_filter(&format, &[ext])
        .set_file_name(format!("transcript.{ext}"))
        .blocking_save_file()
        .and_then(|p| p.as_path().map(|p| p.to_string_lossy().into_owned()));
    serde_json::json!({"path": path})
}
```

- [ ] **Step 5: Wire it up in `lib.rs`**

Replace `crates/asr-memo-app/src/lib.rs` with:

```rust
mod commands;
mod state;

use state::AppState;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            commands::get_readiness,
            commands::prepare,
            commands::start_live,
            commands::stop_session,
            commands::get_transcript,
            commands::export_transcript,
            commands::transcribe_file,
            commands::pick_audio_file,
            commands::pick_export_path,
        ])
        .run(tauri::generate_context!())
        .expect("error while running ASR-Memo");
}
```

- [ ] **Step 6: Fill in `ui/tauri-shim.js`**

```javascript
/* Fabricates window.pywebview.api over Tauri invoke so app.js runs unchanged
 * under both shells. Loaded before app.js; inert under real pywebview. */
(function () {
  if (!window.__TAURI__) return; // running under pywebview — do nothing
  const { invoke } = window.__TAURI__.core;
  const { listen } = window.__TAURI__.event;

  window.pywebview = {
    api: {
      get_readiness: () => invoke("get_readiness"),
      prepare: () => invoke("prepare"),
      start_live: (sources, language_hint = null) =>
        invoke("start_live", { sources, language_hint }),
      stop_session: () => invoke("stop_session"),
      transcribe_file: (path, language_hint = null) =>
        invoke("transcribe_file", { path, language_hint }),
      pick_audio_file: () => invoke("pick_audio_file"),
      pick_export_path: (format) => invoke("pick_export_path", { format }),
      export_transcript: (path, format = null) =>
        invoke("export_transcript", { path, format }),
      get_transcript: () => invoke("get_transcript"),
    },
  };

  listen("backend-event", (e) => {
    if (typeof window.onBackendEvent === "function") window.onBackendEvent(e.payload);
  });

  // app.js boots on this event (see its `pywebviewready` listener + ready-poll).
  window.dispatchEvent(new Event("pywebviewready"));
})();
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cargo test --workspace`
Expected: `test result: ok.` in both crates (7 core + 4 app tests).

- [ ] **Step 8: Manual end-to-end check**

Run: `cargo run -p asr-memo-app`
Expected: readiness passes → main screen; click **Start** → status flips to *recording*, fake segments stream in with **S1/S2** speaker colors; **Stop** → status *stopped*, review screen; export writes a real `.md`/`.json` file.

- [ ] **Step 9: Commit**

```bash
git add crates/asr-memo-app
git commit -m "feat(v2): tauri commands + event bridge, fake-mode end-to-end"
```

---

### Task 6: CI matrix (3 OSes, offline)

**Files:**
- Create: `.github/workflows/rust-ci.yml`

**Interfaces:**
- Consumes: the workspace from Tasks 1–5.
- Produces: green `cargo fmt` / `clippy` / `test` on macOS, Windows, Ubuntu for every push/PR.

- [ ] **Step 1: Write the workflow**

```yaml
name: rust-ci

on:
  push:
    branches: ["**"]
  pull_request:

env:
  CARGO_TERM_COLOR: always

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [macos-14, windows-latest, ubuntu-22.04]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - name: Install Tauri system deps (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y libwebkit2gtk-4.1-dev build-essential \
            libssl-dev libayatana-appindicator3-dev librsvg2-dev
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2
      - name: Format
        run: cargo fmt --all --check
      - name: Clippy
        run: cargo clippy --workspace -- -D warnings
      - name: Test (offline — no models, no audio, no network)
        run: cargo test --workspace
```

- [ ] **Step 2: Verify locally (the parts CI runs)**

Run: `cargo fmt --all --check && cargo clippy --workspace -- -D warnings && cargo test --workspace`
Expected: all three pass. Fix any fmt/clippy findings now (they're part of this task).

- [ ] **Step 3: Commit and push; confirm CI**

```bash
git add .github/workflows/rust-ci.yml
git commit -m "ci(v2): 3-OS rust matrix (fmt, clippy, offline tests)"
git push -u origin HEAD
```

Expected: all three matrix jobs green on GitHub (`gh run watch` to follow).

---

## Self-review notes (done at write time)

- **Spec coverage (Phase-1 slice):** workspace/UI-reuse/DTO-carryover (§4) → Tasks 1, 4, 5; traits+fakes offline discipline (§8) → Task 2; session/live-events skeleton (§5.2 orchestrator interface) → Task 3; CI matrix (§5.3) → Task 6. Audio engine, real inference, SQLite, packaging are later phases per the master plan.
- **Type consistency:** `SegmentDto`/`ReadinessDto`/`ErrorInfoDto` names used identically in Tasks 1/3/5; `SessionEvent` variants match between Tasks 3 and 5; fake constructors (`from_script`/`with_tokens`/`alternating`/`demo_script`) match between Tasks 2 and 5.
- **Known simplifications (deliberate, non-placeholder):** midpoint aligner (real code, replaced in Phase 4), `transcribe_file` returns a typed `not_implemented` error until Phase 3, fake readiness in Phase 1. Speaker palette hexes in Task 5's `COLORS` are copied from the canonical `src/meeting_asr/export/palette.py::SPEAKER_COLORS` (verified at plan-writing time).
