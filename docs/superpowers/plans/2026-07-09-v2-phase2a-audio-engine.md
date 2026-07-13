# ASR-Memo v2 — Phase 2a: macOS Audio Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A real macOS audio engine (mic + system tap → mix → 48 kHz WAV record + health) that fixes the "Teams remote voice not transcribed" bug, integrated behind the Phase-1 `AudioCapture` trait and surfaced with a Record mode, level meters, and a "system silent" warning.

**Architecture:** New `crates/asr-memo-core/src/audio/` modules (offline-testable core: `resample`/`health`/`record`/`mixer`; hardware-gated captures: `mic`/`system/macos`) implement the existing `AudioCapture` trait. The app adds a **capture mode** (`start_capture`/`stop_capture`/`reveal_recording`) distinct from the fake `start_live`, plus an `audio-health` Tauri event and minimal UI (Record button, two level meters, warning banner). No transcript in 2a (ASR is Phase 3).

**Tech Stack:** Rust — `cpal` 0.15 (mic, all OSes), `cidre` git rev `a9587fa` (macOS Core Audio tap, `cfg`-gated), `rubato` 0.15 (resampling). Existing `asr-memo-core`/`asr-memo-app`. macOS 14.4+.

**Spec:** `docs/superpowers/specs/2026-07-09-v2-phase2a-audio-engine-design.md`.

## Global Constraints

- Do NOT modify `app/` or `src/meeting_asr/` (Python reference). UI work edits `crates/asr-memo-app/ui/` (the copy).
- Offline-first: every test runs with no audio hardware, no models, no network. Real `cpal`/`cidre` captures are `#[ignore]` hardware-gated tests run manually.
- 3-OS CI stays green: macOS compiles + runs the real-path unit tests; Windows/Linux compile the workspace (cpal on Linux needs `libasound2-dev`; cidre is `cfg(target_os="macos")` so it's absent on Win/Linux with a non-mac stub for the system source).
- The 16 kHz mono `AudioFrame` (Phase-1 `traits.rs`) is the mixer's output and the existing `Session`'s input — unchanged contract.
- Speaker/DTO/event contracts from Phase 1 are locked. New event type `audio_health` is added alongside `status`/`segment`/`error`.
- `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace` must be green before each commit. Rust is **not** on PATH — `source "$HOME/.cargo/env"` first.
- macOS capture requires the app bundle's `Info.plist` to declare `NSMicrophoneUsageDescription` and `NSAudioCaptureUsageDescription` (Task 10).
- Adapt `cidre` tap code faithfully from meetily's `frontend/src-tauri/src/audio/capture/core_audio.rs` (MIT) — it compiles and works; the reference is the source of truth for the `cidre` API at rev `a9587fa`.

---

### Task 1: Dependencies + `audio/` module scaffolding

**Files:**
- Modify: `crates/asr-memo-core/Cargo.toml`
- Modify: `crates/asr-memo-core/src/lib.rs`
- Create: `crates/asr-memo-core/src/audio/mod.rs`
- Create (empty stubs): `crates/asr-memo-core/src/audio/{resample.rs, health.rs, record.rs, mixer.rs, mic.rs, system.rs}`
- Modify: `.github/workflows/rust-ci.yml` (Linux: add `libasound2-dev` for cpal)

**Interfaces:**
- Consumes: Phase-1 `asr-memo-core`.
- Produces: an `audio` module declared and compiling on all 3 OSes; later tasks populate the submodules.

- [ ] **Step 1: Add dependencies**

`crates/asr-memo-core/Cargo.toml` `[dependencies]` becomes:

```toml
[dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
rubato = "0.15"
cpal = "0.15"

[target.'cfg(target_os = "macos")'.dependencies]
cidre = { git = "https://github.com/yury/cidre", rev = "a9587fa", features = ["av"] }
```

- [ ] **Step 2: Declare the audio module**

`crates/asr-memo-core/src/lib.rs`:

```rust
//! asr-memo-core: platform-independent core for ASR-Memo v2.
//! JSON contracts here are locked to the Python bridge (`app/dto.py`).

pub mod audio;
pub mod fakes;
pub mod session;
pub mod traits;
pub mod types;
```

- [ ] **Step 3: Create the module skeleton**

`crates/asr-memo-core/src/audio/mod.rs`:

```rust
//! Real audio engine (Phase 2a): capture → 50 ms mix → record + health.
//! Offline-testable core first; cpal/cidre captures are hardware-gated.

pub mod health;
pub mod mic;
pub mod mixer;
pub mod record;
pub mod resample;
pub mod system;
```

Create each of `resample.rs`, `health.rs`, `record.rs`, `mixer.rs`, `mic.rs`, `system.rs` with a single header line (e.g. `//! Resampling — implemented in Task 2.`) so the crate compiles.

- [ ] **Step 4: Add the Linux cpal dep to CI**

In `.github/workflows/rust-ci.yml`, the Linux apt install becomes:

```yaml
          sudo apt-get install -y libwebkit2gtk-4.1-dev build-essential \
            libssl-dev libayatana-appindicator3-dev librsvg2-dev libasound2-dev
```

- [ ] **Step 5: Build + test (deps resolve on all platforms, fmt/clippy clean)**

Run: `source "$HOME/.cargo/env" && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: all green (pulls cpal/rubato/cidre; cidre skipped on Win/Linux).

- [ ] **Step 6: Commit**

```bash
git add crates/asr-memo-core/Cargo.toml crates/asr-memo-core/src/audio crates/asr-memo-core/src/lib.rs .github/workflows/rust-ci.yml
git commit -m "feat(v2-2a): audio engine deps + module scaffolding"
```

---

### Task 2: `resample.rs` — persistent rubato resampler

**Files:**
- Modify: `crates/asr-memo-core/src/audio/resample.rs`
- Test: inline `#[cfg(test)] mod tests` in the same file

**Interfaces:**
- Consumes: nothing (leaf utility).
- Produces: `pub struct PersistentResampler` with `fn new(from_rate: u32, to_rate: u32, chunk: usize) -> Self`, `fn process(&mut self, input: &[f32]) -> Vec<f32>` (buffers variable-length input into `chunk`-sized frames), `fn flush(&mut self) -> Vec<f32>` (drains buffered remainder), `fn set_rates(&mut self, from_rate: u32, to_rate: u32)` (reconfigure for a device rate change — used by captures for Bluetooth renegotiation). Used by Tasks 6 and 7.

- [ ] **Step 1: Write the failing tests**

`crates/asr-memo-core/src/audio/resample.rs` (tests first):

```rust
//! One persistent `rubato` resampler per stream. Never recreate per chunk —
//! meetily found that amplifies energy ~173% and yields wrong output sizes.

use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

pub struct PersistentResampler {
    from_rate: u32,
    to_rate: u32,
    chunk: usize,
    resampler: SincFixedIn<f32>,
    buffer: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f32]) -> f64 {
        if x.is_empty() { return 0.0; }
        let s: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s / x.len() as f64).sqrt()
    }

    #[test]
    fn decimates_48k_to_16k_at_correct_ratio() {
        // 480 samples @48k = 10ms → 160 samples @16k.
        let mut r = PersistentResampler::new(48000, 16000, 512);
        let input = vec![0.5f32; 480 * 4]; // 40ms
        let mut out = Vec::new();
        out.extend(r.process(&input));
        out.extend(r.flush());
        let ratio = out.len() as f64 / input.len() as f64;
        assert!((ratio - 1.0 / 3.0).abs() < 0.02, "ratio={ratio}");
    }

    #[test]
    fn preserves_energy_across_variable_chunks() {
        let mut r = PersistentResampler::new(48000, 16000, 512);
        let signal: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.02 * 2.0 * 3.14159).sin() * 0.8).collect();
        let mut out = Vec::new();
        // Feed in awkward chunk sizes that don't divide chunk=512.
        for chunk in [320usize, 512, 1024, 700, 256] {
            let mut i = 0;
            while i < signal.len() {
                let end = (i + chunk).min(signal.len());
                out.extend(r.process(&signal[i..end]));
                i = end;
            }
        }
        out.extend(r.flush());
        let in_rms = rms(&signal);
        let out_rms = rms(&out);
        // Resampling preserves amplitude within a few %.
        assert!((out_rms - in_rms).abs() / in_rms < 0.05, "in={in_rms:.4} out={out_rms:.4}");
    }

    #[test]
    fn set_rates_reconfigures_for_rate_change() {
        let mut r = PersistentResampler::new(48000, 16000, 512);
        let _ = r.process(&vec![0.3f32; 1024]);
        let _ = r.flush();
        // Bluetooth switches output to 44100.
        r.set_rates(44100, 16000);
        let out = r.process(&vec![0.3f32; 44100 / 10]); // 100ms @44100
        assert!(!out.is_empty());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p asr-memo-core audio::resample`
Expected: FAIL — `process`/`new` unimplemented (compile error on the test's calls).

- [ ] **Step 3: Implement `PersistentResampler`**

Add above the tests (replace the placeholder header):

```rust
impl PersistentResampler {
    pub fn new(from_rate: u32, to_rate: u32, chunk: usize) -> Self {
        Self {
            from_rate,
            to_rate,
            chunk,
            resampler: build_resampler(from_rate, to_rate, chunk),
            buffer: Vec::with_capacity(chunk * 2),
        }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.from_rate == self.to_rate {
            return input.to_vec();
        }
        self.buffer.extend_from_slice(input);
        let mut out = Vec::new();
        while self.buffer.len() >= self.chunk {
            let frame: Vec<f32> = self.buffer.drain(..self.chunk).collect();
            let waves = vec![frame];
            if let Ok(mut resampled) = self.resampler.process(&waves, None) {
                if let Some(ch) = resampled.get_mut(0) {
                    out.append(ch);
                }
            }
        }
        out
    }

    pub fn flush(&mut self) -> Vec<f32> {
        if self.from_rate == self.to_rate {
            return std::mem::take(&mut self.buffer);
        }
        let remainder = std::mem::take(&mut self.buffer);
        if remainder.is_empty() {
            return Vec::new();
        }
        // Pad the final partial chunk with zeros to flush the resampler.
        let mut padded = remainder;
        padded.resize(self.chunk, 0.0);
        let waves = vec![padded];
        match self.resampler.process(&waves, None) {
            Ok(mut resampled) => resampled.pop().unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    pub fn set_rates(&mut self, from_rate: u32, to_rate: u32) {
        self.from_rate = from_rate;
        self.to_rate = to_rate;
        self.buffer.clear();
        self.resampler = build_resampler(from_rate, to_rate, self.chunk);
    }
}

fn build_resampler(from_rate: u32, to_rate: u32, chunk: usize) -> SincFixedIn<f32> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    SincFixedIn::<f32>::new(to_rate as f64 / from_rate as f64, 2.0, params, chunk, 1).unwrap()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p asr-memo-core audio::resample`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/asr-memo-core/src/audio/resample.rs
git commit -m "feat(v2-2a): persistent rubato resampler with rate-change support"
```

---

### Task 3: `health.rs` — per-source levels + silence/rate flags

**Files:**
- Modify: `crates/asr-memo-core/src/audio/health.rs`
- Test: inline

**Interfaces:**
- Consumes: nothing.
- Produces: `pub struct SourceHealth { pub rms: f32, pub peak: f32 }`, `pub struct HealthSnapshot { pub mic: SourceHealth, pub system: SourceHealth, pub flags: Vec<HealthFlag> }`, `pub enum HealthFlag { SystemSilent, RateChanged }`, `pub struct HealthMonitor { ... }` with `fn new(silence_threshold: f32, silence_window_frames: usize)`, `fn update(&mut self, mic_window: &[f32], system_window: &[f32], system_active: bool) -> HealthSnapshot`. Used by Task 5 (mixer).

- [ ] **Step 1: Write the failing tests**

`crates/asr-memo-core/src/audio/health.rs`:

```rust
//! Per-source RMS/peak + sustained-silence detection (the permission-denied
//! signature: a macOS process tap returns all-zeros when permission is missing).

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceHealth {
    pub rms: f32,
    pub peak: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthFlag {
    SystemSilent,
    RateChanged,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HealthSnapshot {
    pub mic: SourceHealth,
    pub system: SourceHealth,
    pub flags: Vec<HealthFlag>,
}

pub struct HealthMonitor {
    silence_threshold: f32,
    silence_window_frames: usize,
    consecutive_silent: usize,
    rate_changed_pending: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_rms_and_peak() {
        let mut m = HealthMonitor::new(0.001, 5);
        let snap = m.update(&[0.5, -0.5, 0.25], &[0.0, 0.0, 0.0], true);
        assert!((snap.mic.rms - 0.408).abs() < 0.01);
        assert_eq!(snap.mic.peak, 0.5);
    }

    #[test]
    fn flags_system_silent_only_after_sustained_zeros() {
        let mut m = HealthMonitor::new(0.001, 3);
        for _ in 0..2 {
            let s = m.update(&[0.4; 64], &[0.0; 64], true);
            assert!(!s.flags.contains(&HealthFlag::SystemSilent));
        }
        let s = m.update(&[0.4; 64], &[0.0; 64], true); // 3rd silent window
        assert!(s.flags.contains(&HealthFlag::SystemSilent));
    }

    #[test]
    fn non_silent_system_resets_silence_counter() {
        let mut m = HealthMonitor::new(0.001, 3);
        for _ in 0..2 { m.update(&[0.4; 64], &[0.0; 64], true); }
        m.update(&[0.4; 64], &[0.1; 64], true); // real signal resets
        let s = m.update(&[0.4; 64], &[0.0; 64], true);
        assert!(!s.flags.contains(&HealthFlag::SystemSilent)); // counter restarted
    }

    #[test]
    fn system_inactive_never_flags_silent() {
        let mut m = HealthMonitor::new(0.001, 2);
        for _ in 0..5 {
            let s = m.update(&[0.4; 64], &[0.0; 64], false);
            assert!(!s.flags.contains(&HealthFlag::SystemSilent));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p asr-memo-core audio::health`
Expected: FAIL — `HealthMonitor::new`/`update` unimplemented.

- [ ] **Step 3: Implement**

```rust
impl HealthMonitor {
    pub fn new(silence_threshold: f32, silence_window_frames: usize) -> Self {
        Self { silence_threshold, silence_window_frames, consecutive_silent: 0, rate_changed_pending: false }
    }

    /// Mark that the device sample rate changed; the next snapshot reports
    /// `RateChanged` once and clears the pending flag.
    pub fn notify_rate_changed(&mut self) {
        self.rate_changed_pending = true;
    }

    pub fn update(&mut self, mic_window: &[f32], system_window: &[f32], system_active: bool) -> HealthSnapshot {
        let mic = source_health(mic_window);
        let system = source_health(system_window);
        let mut flags = Vec::new();

        if system_active && system.rms <= self.silence_threshold {
            self.consecutive_silent += 1;
            if self.consecutive_silent >= self.silence_window_frames {
                flags.push(HealthFlag::SystemSilent);
            }
        } else {
            self.consecutive_silent = 0;
        }
        if self.rate_changed_pending {
            flags.push(HealthFlag::RateChanged);
            self.rate_changed_pending = false;
        }
        HealthSnapshot { mic, system, flags }
    }
}

fn source_health(window: &[f32]) -> SourceHealth {
    if window.is_empty() {
        return SourceHealth { rms: 0.0, peak: 0.0 };
    }
    let sum_sq: f64 = window.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms = (sum_sq / window.len() as f64).sqrt() as f32;
    let peak = window.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    SourceHealth { rms, peak }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p asr-memo-core audio::health`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/asr-memo-core/src/audio/health.rs
git commit -m "feat(v2-2a): audio health monitor (levels + system-silent/rate flags)"
```

---

### Task 4: `record.rs` — incremental 48 kHz WAV writer

**Files:**
- Modify: `crates/asr-memo-core/src/audio/record.rs`
- Test: inline

**Interfaces:**
- Consumes: nothing.
- Produces: `pub struct WavWriter { ... }` with `fn create(path: &Path, sample_rate: u32) -> std::io::Result<Self>`, `fn write_samples(&mut self, pcm: &[f32]) -> std::io::Result<()>` (converts f32→i16, appends), `fn close(self) -> std::io::Result<()>` (patches the RIFF/data lengths so a file closed mid-session is valid). Used by Task 8.

- [ ] **Step 1: Write the failing tests**

`crates/asr-memo-core/src/audio/record.rs`:

```rust
//! Incremental 48 kHz WAV writer. Header written on open; PCM appended per
//! window; lengths patched on close so a crash leaves a valid (shorter) file.

use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;

pub struct WavWriter {
    file: File,
    data_bytes: u32,
    sample_rate: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u32(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }

    #[test]
    fn writes_valid_wav_header_and_patches_lengths_on_close() {
        let tmp = tempfile();
        let path = std::path::Path::new(&tmp);
        {
            let mut w = WavWriter::create(path, 48000).unwrap();
            w.write_samples(&[0.0, 0.5, -0.5, 1.0]).unwrap();
            w.close().unwrap();
        }
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(read_u32(&bytes, 16), 16);        // PCM fmt chunk size
        assert_eq!(read_u32(&bytes, 24), 48000);     // sample rate
        assert_eq!(read_u32(&bytes, 32), 48000 * 2); // byte rate (mono s16)
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(read_u32(&bytes, 40), 8);         // 4 samples * 2 bytes
        assert_eq!(bytes.len(), 44 + 8);
    }

    #[test]
    fn drop_without_close_still_leaves_valid_header() {
        // If the process is killed, the file on disk has the open-time header
        // (data length 0). It's a valid WAV with whatever was flushed by the OS.
        let tmp = tempfile();
        let path = std::path::Path::new(&tmp);
        {
            let mut w = WavWriter::create(path, 48000).unwrap();
            w.write_samples(&[0.25, -0.25]).unwrap();
            // intentionally no close() — flush what we can
            let _ = w.file.sync_all();
        }
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert!(bytes.len() >= 44);
    }

    fn tempfile() -> String {
        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.join(format!("asr-memo-test-{nanos}.wav")).to_string_lossy().into_owned()
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p asr-memo-core audio::record`
Expected: FAIL — `WavWriter::create` unimplemented.

- [ ] **Step 3: Implement**

```rust
impl WavWriter {
    pub fn create(path: &Path, sample_rate: u32) -> io::Result<Self> {
        let mut file = File::create(path)?;
        // Open-time header with data length 0; patched on close.
        file.write_all(b"RIFF")?;
        file.write_all(&0u32.to_le_bytes())?; // file size placeholder
        file.write_all(b"WAVE")?;
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?;            // PCM
        file.write_all(&1u16.to_le_bytes())?;            // mono
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
        file.write_all(&2u16.to_le_bytes())?;            // block align
        file.write_all(&16u16.to_le_bytes())?;           // bits per sample
        file.write_all(b"data")?;
        file.write_all(&0u32.to_le_bytes())?; // data length placeholder
        file.sync_all()?;
        Ok(Self { file, data_bytes: 0, sample_rate })
    }

    pub fn write_samples(&mut self, pcm: &[f32]) -> io::Result<()> {
        let mut bytes = Vec::with_capacity(pcm.len() * 2);
        for &s in pcm {
            let clamped = s.clamp(-1.0, 1.0);
            let i16_sample = (clamped * 32767.0) as i16;
            bytes.extend_from_slice(&i16_sample.to_le_bytes());
        }
        self.file.write_all(&bytes)?;
        self.data_bytes = self.data_bytes.saturating_add(bytes.len() as u32);
        Ok(())
    }

    pub fn close(mut self) -> io::Result<()> {
        self.file.flush()?;
        // Patch RIFF file size (everything after "RIFF") and data length.
        let riff_size = 36 + self.data_bytes;
        self.file.seek(SeekFrom::Start(4))?;
        self.file.write_all(&riff_size.to_le_bytes())?;
        self.file.seek(SeekFrom::Start(40))?;
        self.file.write_all(&self.data_bytes.to_le_bytes())?;
        self.file.flush()?;
        self.file.sync_all()?;
        Ok(())
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p asr-memo-core audio::record`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/asr-memo-core/src/audio/record.rs
git commit -m "feat(v2-2a): incremental WAV writer (length-patched on close)"
```

---

### Task 5: `mixer.rs` — `MixedCapture` (50 ms align, RMS dynamic mix, soft-scale, health)

**Files:**
- Modify: `crates/asr-memo-core/src/audio/mixer.rs`
- Test: inline

**Interfaces:**
- Consumes: `crate::traits::{AudioCapture, AudioFrame, CoreError}`, `crate::types::AudioSource`, `crate::audio::health::{HealthMonitor, HealthSnapshot}`.
- Produces: `pub struct MixedCapture { ... }` implementing `AudioCapture`. Constructor: `fn new(mic: Box<dyn AudioCapture>, system: Option<Box<dyn AudioCapture>>, on_health: Box<dyn Fn(HealthSnapshot) + Send>) -> Self`. `start` spawns the two sub-captures and a mixer thread that aligns 50 ms windows (800 samples @16 kHz), computes RMS ratios clamped 10–90%, soft-scales sums >1.0, emits mixed `AudioFrame`s (source = dominant) and pushes health snapshots. `system_active` = whether a system capture was supplied. Used by Task 8.

> Design note: the mixer runs on its own thread (not in the audio callback). Sub-captures push 16 kHz mono frames onto per-source ring buffers; the mixer drains aligned 50 ms windows. This keeps allocation out of the capture callback.

- [ ] **Step 1: Write the failing tests**

`crates/asr-memo-core/src/audio/mixer.rs`:

```rust
//! Aligns mic + system 16 kHz streams in 50 ms windows, applies RMS-based
//! dynamic mixing (ratios 10–90%) with soft-scaling, emits mixed frames +
//! health snapshots. Tested with fake sub-captures (no cpal/cidre).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use crate::audio::health::{HealthMonitor, HealthSnapshot, HealthFlag};
use crate::traits::{AudioCapture, AudioFrame, CoreError};
use crate::types::AudioSource;

const RATE: u32 = 16000;
const WINDOW_SECS: f64 = 0.05;
const WINDOW_SAMPLES: usize = (RATE as f64 * WINDOW_SECS) as usize; // 800
const MIN_RATIO: f32 = 0.1;
const MAX_RATIO: f32 = 0.9;

pub struct MixedCapture {
    mic: Option<Box<dyn AudioCapture>>,
    system: Option<Box<dyn AudioCapture>>,
    on_health: Box<dyn Fn(HealthSnapshot) + Send>,
    stop: Arc<AtomicBool>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{AudioCapture, AudioFrame};
    use crate::types::AudioSource;
    use std::sync::mpsc;

    /// A fake capture that emits a scripted sequence of 50 ms frames then ends.
    struct ScriptCapture { frames: Vec<AudioFrame> }
    impl AudioCapture for ScriptCapture {
        fn start(&mut self, mut on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError> {
            let frames: Vec<AudioFrame> = self.frames.drain(..).collect();
            std::thread::spawn(move || for f in frames { on_frame(f); });
            Ok(())
        }
        fn stop(&mut self) {}
    }

    fn win(t0: f64, level: f32, src: AudioSource) -> AudioFrame {
        AudioFrame { pcm: vec![level; WINDOW_SAMPLES], t_start: t0, t_end: t0 + WINDOW_SECS, source: src }
    }

    #[test]
    fn emits_mixed_frames_aligned_to_50ms_and_reports_health() {
        let mic: Box<dyn AudioCapture> = Box::new(ScriptCapture {
            frames: vec![win(0.0, 0.4, AudioSource::Microphone), win(0.05, 0.4, AudioSource::Microphone)],
        });
        let sys: Box<dyn AudioCapture> = Box::new(ScriptCapture {
            frames: vec![win(0.0, 0.2, AudioSource::System), win(0.05, 0.0, AudioSource::System)],
        });
        let (tx, rx) = mpsc::channel::<AudioFrame>();
        let (htx, hrx) = mpsc::channel::<HealthSnapshot>();
        let mut mix = MixedCapture::new(Some(mic), Some(sys), Box::new(move |h| htx.send(h).unwrap()));
        mix.start(Box::new(move |f| tx.send(f).unwrap())).unwrap();
        // give the mixer thread time to drain
        std::thread::sleep(std::time::Duration::from_millis(200));
        mix.stop();

        let frames: Vec<AudioFrame> = rx.try_iter().collect();
        assert!(frames.len() >= 2, "expected >=2 mixed windows, got {}", frames.len());
        for f in &frames {
            assert_eq!(f.pcm.len(), WINDOW_SAMPLES);
            assert!((f.t_end - f.t_start - WINDOW_SECS).abs() < 1e-6);
        }
        // Mic louder than system on window 0 → dominant source is Microphone.
        assert_eq!(frames[0].source, AudioSource::Microphone);
        // Health snapshots were delivered.
        let healths: Vec<HealthSnapshot> = hrx.try_iter().collect();
        assert!(!healths.is_empty());
    }

    #[test]
    fn soft_scales_when_sum_exceeds_unity() {
        // Two loud windows summed naively would clip; soft-scale keeps |x|<=1.
        let mic: Box<dyn AudioCapture> = Box::new(ScriptCapture { frames: vec![win(0.0, 0.9, AudioSource::Microphone)] });
        let sys: Box<dyn AudioCapture> = Box::new(ScriptCapture { frames: vec![win(0.0, 0.9, AudioSource::System)] });
        let (tx, rx) = mpsc::channel::<AudioFrame>();
        let mut mix = MixedCapture::new(Some(mic), Some(sys), Box::new(|_| {}));
        mix.start(Box::new(move |f| tx.send(f).unwrap())).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(150));
        mix.stop();
        let frames: Vec<AudioFrame> = rx.try_iter().collect();
        let max_abs = frames.iter().flat_map(|f| f.pcm.iter().copied()).map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(max_abs <= 1.0 + 1e-6, "clipped: {max_abs}");
    }

    #[test]
    fn system_silence_is_flagged() {
        let mic: Box<dyn AudioCapture> = Box::new(ScriptCapture {
            frames: (0..6).map(|i| win(i as f64 * 0.05, 0.4, AudioSource::Microphone)).collect(),
        });
        let sys: Box<dyn AudioCapture> = Box::new(ScriptCapture {
            frames: (0..6).map(|i| win(i as f64 * 0.05, 0.0, AudioSource::System)).collect(),
        });
        let (htx, hrx) = mpsc::channel::<HealthSnapshot>();
        let mut mix = MixedCapture::new(Some(mic), Some(sys), Box::new(move |h| htx.send(h).unwrap()));
        mix.start(Box::new(|_| {})).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(250));
        mix.stop();
        let healths: Vec<HealthSnapshot> = hrx.try_iter().collect();
        assert!(healths.iter().any(|h| h.flags.contains(&HealthFlag::SystemSilent)));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p asr-memo-core audio::mixer`
Expected: FAIL — `MixedCapture::new` unimplemented.

- [ ] **Step 3: Implement**

```rust
impl MixedCapture {
    pub fn new(
        mic: Option<Box<dyn AudioCapture>>,
        system: Option<Box<dyn AudioCapture>>,
        on_health: Box<dyn Fn(HealthSnapshot) + Send>,
    ) -> Self {
        Self { mic, system, on_health, stop: Arc::new(AtomicBool::new(false)), handles: Vec::new() }
    }

    fn rms(window: &[f32]) -> f32 {
        if window.is_empty() { return 0.0; }
        let s: f64 = window.iter().map(|&x| (x as f64) * (x as f64)).sum();
        (s / window.len() as f64).sqrt() as f32
    }

    fn mix_window(mic_w: &[f32], sys_w: &[f32]) -> Vec<f32> {
        let mic_r = Self::rms(mic_w);
        let sys_r = Self::rms(sys_w);
        let (mr, sr) = dynamic_ratios(mic_r, sys_r);
        let max_len = mic_w.len().max(sys_w.len());
        let mut mixed = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let m = if i < mic_w.len() { mic_w[i] * mr } else { 0.0 };
            let s = if i < sys_w.len() { sys_w[i] * sr } else { 0.0 };
            mixed.push(m + s);
        }
        soft_scale(&mut mixed);
        mixed
    }
}

fn dynamic_ratios(mic_r: f32, sys_r: f32) -> (f32, f32) {
    if mic_r == 0.0 && sys_r == 0.0 { return (0.5, 0.5); }
    if mic_r == 0.0 { return (0.0, 1.0); }
    if sys_r == 0.0 { return (1.0, 0.0); }
    let total = mic_r + sys_r;
    let mr = ((mic_r / total).clamp(MIN_RATIO, MAX_RATIO));
    (mr, 1.0 - mr)
}

fn soft_scale(mixed: &mut [f32]) {
    let peak = mixed.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1.0 {
        let g = 1.0 / peak;
        for s in mixed.iter_mut() { *s *= g; }
    }
}

impl AudioCapture for MixedCapture {
    fn start(&mut self, mut on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError> {
        let stop = self.stop.clone();
        let (mic_tx, mic_rx) = mpsc::channel::<AudioFrame>();
        let (sys_tx, sys_rx) = mpsc::channel::<AudioFrame>();
        let system_active = self.system.is_some();

        if let Some(mut mic) = self.mic.take() {
            let tx = mic_tx.clone();
            mic.start(Box::new(move |f| { let _ = tx.send(f); }))?;
            // mic.stop() is never called per-capture; MixedCapture::stop stops the loop.
            std::mem::forget(mic); // keep the capture alive for the session lifetime
        }
        if let Some(mut sys) = self.system.take() {
            let tx = sys_tx.clone();
            sys.start(Box::new(move |f| { let _ = tx.send(f); }))?;
            std::mem::forget(sys);
        }

        let on_health = take_boxed(&mut self.on_health);
        self.handles.push(std::thread::spawn(move || {
            let mut health = HealthMonitor::new(0.001, 5);
            let mut mic_buf: Vec<f32> = Vec::new();
            let mut sys_buf: Vec<f32> = Vec::new();
            let mut t = 0.0f64;
            loop {
                if stop.load(Ordering::Acquire) { break; }
                // Drain newly arrived frames into the buffers.
                while let Ok(f) = mic_rx.try_recv() { mic_buf.extend(f.pcm); }
                while let Ok(f) = sys_rx.try_recv() { sys_buf.extend(f.pcm); }
                if mic_buf.len() >= WINDOW_SAMPLES {
                    let mic_w: Vec<f32> = mic_buf.drain(..WINDOW_SAMPLES).collect();
                    let sys_w: Vec<f32> = if sys_buf.len() >= WINDOW_SAMPLES {
                        sys_buf.drain(..WINDOW_SAMPLES).collect()
                    } else {
                        vec![0.0; WINDOW_SAMPLES.min(sys_buf.len().max(0))]
                    };
                    let snap = health.update(&mic_w, &sys_w, system_active);
                    on_health(snap);
                    let dominant = if Self::rms(&mic_w) >= Self::rms(&sys_w) { AudioSource::Microphone } else { AudioSource::System };
                    let pcm = Self::mix_window(&mic_w, &sys_w);
                    on_frame(AudioFrame { pcm, t_start: t, t_end: t + WINDOW_SECS, source: dominant });
                    t += WINDOW_SECS;
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        }));
        Ok(())
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        for h in self.handles.drain(..) { let _ = h.join(); }
    }
}

fn take_boxed<T>(slot: &mut Box<T>) -> Box<T> {
    let taken = std::mem::replace(slot as &mut T, unsafe { std::ptr::read(slot.as_ref()) as *mut T });
    unsafe { Box::from_raw(taken) }
}
```

> Note to implementer: `take_boxed` above is awkward; the clean version is to move `on_health` into `start` by value. Since `start(&mut self)` takes `&mut self`, store `on_health: Option<Box<dyn Fn(HealthSnapshot) + Send>>` and `.take().unwrap()` inside `start`. Apply that simplification — the test only needs the closure invoked.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p asr-memo-core audio::mixer`
Expected: 3 passed. (If the `take_boxed` wart fails clippy, switch to the `Option<Box<...>>` field noted above.)

- [ ] **Step 5: Commit**

```bash
git add crates/asr-memo-core/src/audio/mixer.rs
git commit -m "feat(v2-2a): MixedCapture — 50ms align, RMS dynamic mix, soft-scale, health"
```

---

### Task 6: `mic.rs` — `CpalMicrophoneCapture`

**Files:**
- Modify: `crates/asr-memo-core/src/audio/mic.rs`
- Test: inline (`#[ignore]`)

**Interfaces:**
- Consumes: `crate::traits::{AudioCapture, AudioFrame, CoreError}`, `crate::types::AudioSource`, `crate::audio::resample::PersistentResampler`.
- Produces: `pub struct CpalMicrophoneCapture` implementing `AudioCapture`. Constructor `fn new() -> Self`. The cpal callback reads the device's nominal sample rate, resamples to 16 kHz via a `PersistentResampler` (reconfigured on rate change), and pushes `AudioFrame`s onto a channel; a forwarding thread calls `on_frame`. Real hardware — `#[ignore]` test only.

- [ ] **Step 1: Write the `#[ignore]` hardware test**

`crates/asr-memo-core/src/audio/mic.rs`:

```rust
//! Microphone capture via cpal. The cpal callback resamples the device's native
//! rate to 16 kHz mono and forwards AudioFrames. Hardware-gated (#[ignore]).

use std::sync::mpsc;

use crate::audio::resample::PersistentResampler;
use crate::traits::{AudioCapture, AudioFrame, CoreError};
use crate::types::AudioSource;

const OUT_RATE: u32 = 16000;

pub struct CpalMicrophoneCapture {
    _phantom: (),
}

impl CpalMicrophoneCapture {
    pub fn new() -> Self { Self { _phantom: () } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test][ignore] // requires a real microphone + mic permission
    fn captures_two_seconds_of_mic_audio() {
        let mut cap = CpalMicrophoneCapture::new();
        let (tx, rx) = mpsc::channel::<AudioFrame>();
        cap.start(Box::new(move |f| tx.send(f).unwrap())).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(2));
        cap.stop();
        let frames: Vec<AudioFrame> = rx.try_iter().collect();
        assert!(!frames.is_empty());
        let samples: usize = frames.iter().map(|f| f.pcm.len()).sum();
        assert!(samples >= 16000, "expected >=1s of 16kHz audio, got {samples} samples");
    }
}
```

- [ ] **Step 2: Run test to verify it's ignored (compiles)**

Run: `cargo test -p asr-memo-core audio::mic`
Expected: `0 passed; 1 ignored`.

- [ ] **Step 3: Implement the capture**

```rust
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat};
use std::sync::{Arc, Mutex};

pub struct CpalMicrophoneCapture {
    stream: Option<cpal::Stream>,
}

impl CpalMicrophoneCapture {
    pub fn new() -> Self { Self { stream: None } }
}

impl AudioCapture for CpalMicrophoneCapture {
    fn start(&mut self, on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError> {
        let host = cpal::default_host();
        let device = host.default_input_device()
            .ok_or_else(|| CoreError::Capture { code: "mic.no_device".into(), message: "no default input device".into() })?;
        let config = device.default_input_config()
            .map_err(|e| CoreError::Capture { code: "mic.config".into(), message: e.to_string() })?;
        let in_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let fmt = config.sample_format();

        let resampler = Arc::new(Mutex::new(PersistentResampler::new(in_rate, OUT_RATE, 512)));
        let frames_tx = Arc::new(Mutex::new(Box::new(on_frame)));
        let mut emitted: usize = 0;
        let err_cb = |e: cpal::StreamError| eprintln!("mic stream error: {e}");

        let stream = match fmt {
            SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| on_pcm(data, channels, &resampler, &frames_tx, &mut emitted),
                err_cb, None),
            SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _| on_pcm(data, channels, &resampler, &frames_tx, &mut emitted),
                err_cb, None),
            _ => return Err(CoreError::Capture { code: "mic.format".into(), message: "unsupported sample format".into() }),
        }.map_err(|e| CoreError::Capture { code: "mic.build".into(), message: e.to_string() })?;
        stream.play().map_err(|e| CoreError::Capture { code: "mic.play".into(), message: e.to_string() })?;
        self.stream = Some(stream);
        Ok(())
    }

    fn stop(&mut self) {
        if let Some(s) = self.stream.take() {
            // dropping the cpal::Stream stops capture
            drop(s);
        }
    }
}

fn on_pcm<T: Sample + Into<f32>>(
    data: &[T], channels: usize,
    resampler: &Arc<Mutex<PersistentResampler>>,
    frames_tx: &Arc<Mutex<Box<dyn FnMut(AudioFrame) + Send>>>,
    emitted: &mut usize,
) {
    // Downmix to mono f32.
    let mono: Vec<f32> = data
        .chunks_exact(channels.max(1))
        .map(|ch| ch.iter().map(|s| s.into() as f32).sum::<f32>() / channels.max(1) as f32)
        .collect();
    let resampled = resampler.lock().unwrap().process(&mono);
    if resampled.is_empty() { return; }
    let t_start = *emitted as f64 / OUT_RATE as f64;
    let t_end = (*emitted + resampled.len()) as f64 / OUT_RATE as f64;
    *emitted += resampled.len();
    let frame = AudioFrame { pcm: resampled, t_start, t_end, source: AudioSource::Microphone };
    if let Ok(mut cb) = frames_tx.lock() { cb(frame); }
}
```

- [ ] **Step 4: Build (clippy clean; the `#[ignore]` test still ignored)**

Run: `cargo clippy -p asr-memo-core --all-targets -- -D warnings && cargo test -p asr-memo-core audio::mic`
Expected: clippy clean; `0 passed; 1 ignored`.

- [ ] **Step 5: Commit**

```bash
git add crates/asr-memo-core/src/audio/mic.rs
git commit -m "feat(v2-2a): cpal microphone capture (hardware-gated)"
```

---

### Task 7: `system/macos.rs` — `CoreAudioTapCapture` (the bug fix)

**Files:**
- Modify: `crates/asr-memo-core/src/audio/system.rs` (rename to `system/mod.rs` dispatch; create `system/macos.rs`)
- Create: `crates/asr-memo-core/src/audio/system/macos.rs`
- Test: inline (`#[ignore]`)

**Interfaces:**
- Consumes: `crate::traits::{AudioCapture, AudioFrame, CoreError}`, `crate::types::AudioSource`, `crate::audio::resample::PersistentResampler`, `cidre` (macOS).
- Produces: `pub struct CoreAudioTapCapture` implementing `AudioCapture` (cfg macos). Constructor `fn new() -> Self`. Creates a `cidre` **global mono process tap** wrapped in a **tap-only private aggregate device** (never tap + output device together), reads the device's nominal sample rate each callback cycle (reconfiguring the resampler on change → Bluetooth fix), resamples to 16 kHz mono, forwards `AudioFrame`s. Non-macOS: a stub constructor returning `CoreError::Capture`. Faithful port of meetily's `frontend/src-tauri/src/audio/capture/core_audio.rs` (MIT).

> **Implementer note:** the `cidre` API below mirrors meetily's `core_audio.rs` at rev `a9587fa`. If any `cidre` symbol differs, cross-check against meetily's file (the reference of record) and adjust minimally — do not invent.

- [ ] **Step 1: Split `system.rs` into a module dir**

Move the `system.rs` placeholder content into `system/mod.rs` and add the macOS submodule. `crates/asr-memo-core/src/audio/system/mod.rs`:

```rust
//! System-audio capture, per-OS. macOS uses a cidre Core Audio global tap;
//! Windows/Linux land in Phase 2b.

#[cfg(target_os = "macos")]
pub mod macos;
```

(Delete the old `system.rs` placeholder file since it's now a directory.)

- [ ] **Step 2: Write the `#[ignore]` hardware test**

`crates/asr-memo-core/src/audio/system/macos.rs`:

```rust
//! macOS system-audio capture via a cidre Core Audio global mono process tap,
//! wrapped in a tap-only private aggregate device (never tap + output device —
//! that caused meetily's echo). Reads the device sample rate each cycle so a
//! Bluetooth profile switch reconfigures the resampler (the rate-tracking fix).
//! Faithful port of meetily's audio/capture/core_audio.rs (MIT).

#![cfg(target_os = "macos")]

use std::sync::mpsc;

use crate::audio::resample::PersistentResampler;
use crate::traits::{AudioCapture, AudioFrame, CoreError};
use crate::types::AudioSource;

const OUT_RATE: u32 = 16000;

pub struct CoreAudioTapCapture;

impl CoreAudioTapCapture {
    pub fn new() -> Self { CoreAudioTapCapture }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // requires macOS 14.4+, audio-capture permission, and audio playing
    fn captures_two_seconds_of_system_audio() {
        let mut cap = CoreAudioTapCapture::new();
        let (tx, rx) = mpsc::channel::<AudioFrame>();
        cap.start(Box::new(move |f| tx.send(f).unwrap())).unwrap();
        std::thread::sleep(std::time::Duration::from_secs(2));
        cap.stop();
        let frames: Vec<AudioFrame> = rx.try_iter().collect();
        assert!(!frames.is_empty());
        let samples: usize = frames.iter().map(|f| f.pcm.len()).sum();
        assert!(samples >= 16000);
    }
}
```

- [ ] **Step 3: Implement the tap (port of meetily's core_audio.rs)**

Add below the test, still under `#![cfg(target_os = "macos")]`. This creates the global tap + tap-only aggregate, an IOProc that reads PCM, downmixes to mono, resamples (reconfigured on rate change), and forwards frames:

```rust
use cidre::{arc, cf, core_audio as ca, os};
use std::sync::{Arc, Mutex};

// State shared with the Core Audio IOProc callback.
struct TapCtx {
    resampler: Mutex<PersistentResampler>,
    last_rate: Mutex<u32>,
    on_frame: Arc<Mutex<Box<dyn FnMut(AudioFrame) + Send>>>,
    emitted: Mutex<usize>,
}

impl AudioCapture for CoreAudioTapCapture {
    fn start(&mut self, on_frame: Box<dyn FnMut(AudioFrame) + Send>) -> Result<(), CoreError> {
        // 1) Default output device + its UID (the sub-device of the aggregate).
        let output = ca::System::default_output_device()
            .map_err(|e| CoreError::Capture { code: "tap.output".into(), message: format!("{e:?}") })?;
        let output_uid = output.uid()
            .map_err(|e| CoreError::Capture { code: "tap.uid".into(), message: format!("{e:?}") })?;

        // 2) Global mono process tap, excluding no processes.
        let tap_desc = ca::TapDesc::with_mono_global_tap_excluding_processes(&cidre::ns::Array::new());
        let tap = tap_desc.create_process_tap()
            .map_err(|e| CoreError::Capture { code: "tap.create".into(), message: format!("{e:?}") })?;
        let tap_asbd = tap.asbd().map_err(|e| CoreError::Capture { code: "tap.asbd".into(), message: format!("{e:?}") })?;
        let in_rate = tap_asbd.sample_rate as u32;

        // 3) Tap-only aggregate: sub_device = output UID, tap_list = [tap]. Do NOT
        //    also list the output device as a second sub_device (echo).
        let sub_tap = cf::DictionaryOf::with_keys_values(
            &[ca::sub_device_keys::uid()],
            &[tap.uid().unwrap_or_else(|| cf::Uuid::new().to_cf_string()).as_type_ref()],
        );
        let agg_desc = cf::DictionaryOf::with_keys_values(
            &[ca::aggregate_device_keys::is_private(),
              ca::aggregate_device_keys::is_stacked(),
              ca::aggregate_device_keys::tap_auto_start(),
              ca::aggregate_device_keys::name(),
              ca::aggregate_device_keys::main_sub_device(),
              ca::aggregate_device_keys::uid(),
              ca::aggregate_device_keys::tap_list()],
            &[cf::Boolean::value_true().as_type_ref(),
              cf::Boolean::value_false(),
              cf::Boolean::value_true(),
              cf::str!(c"asr-memo-audio-tap").as_type_ref(),
              &output_uid,
              &cf::Uuid::new().to_cf_string(),
              &cf::ArrayOf::from_slice(&[sub_tap.as_ref()])],
        );
        let agg = ca::AggregateDevice::with_desc(&agg_desc)
            .map_err(|e| CoreError::Capture { code: "tap.aggregate".into(), message: format!("{e:?}") })?;

        // 4) Shared context for the IOProc.
        let ctx = Box::new(TapCtx {
            resampler: Mutex::new(PersistentResampler::new(in_rate, OUT_RATE, 512)),
            last_rate: Mutex::new(in_rate),
            on_frame: Arc::new(Mutex::new(on_frame)),
            emitted: Mutex::new(0),
        });
        let ctx_ptr = Box::into_raw(ctx) as *mut TapCtx;

        extern "C" fn io_proc(
            device: ca::Device,
            _now: &cidre::cat::AudioTimeStamp,
            in_data: &cidre::cat::AudioBufList<1>,
            _out_in: &mut cidre::cat::AudioBufList<1>,
            _out_time: &cidre::cat::AudioTimeStamp,
            ctx: Option<&mut TapCtx>,
        ) -> os::Status {
            let Some(ctx) = ctx else { return os::Status::NO_ERR };
            // Rate tracking: reconfigure resampler if the device rate changed.
            if let Ok(rate) = device.nominal_sample_rate() {
                let rate = rate as u32;
                let mut last = ctx.last_rate.lock().unwrap();
                if *last != rate {
                    ctx.resampler.lock().unwrap().set_rates(rate, OUT_RATE);
                    *last = rate;
                }
            }
            // Read interleaved f32 from the first buffer (tap is mono global).
            let buf = &in_data.buffers[0];
            let n = buf.data_bytes_size as usize / std::mem::size_of::<f32>();
            if n == 0 || buf.data.is_null() { return os::Status::NO_ERR; }
            let samples = unsafe { std::slice::from_raw_parts(buf.data as *const f32, n) };
            let resampled = ctx.resampler.lock().unwrap().process(samples);
            if resampled.is_empty() { return os::Status::NO_ERR; }
            let mut emitted = ctx.emitted.lock().unwrap();
            let t_start = *emitted as f64 / OUT_RATE as f64;
            let t_end = (*emitted + resampled.len()) as f64 / OUT_RATE as f64;
            *emitted += resampled.len();
            drop(emitted);
            let frame = AudioFrame { pcm: resampled, t_start, t_end, source: AudioSource::System };
            if let Ok(mut cb) = ctx.on_frame.lock() { cb(frame); }
            os::Status::NO_ERR
        }

        let proc_id = agg.create_io_proc_id(io_proc, Some(unsafe { &mut *ctx_ptr }))
            .map_err(|e| CoreError::Capture { code: "tap.ioproc".into(), message: format!("{e:?}") })?;
        let started = ca::device_start(agg, Some(proc_id))
            .map_err(|e| CoreError::Capture { code: "tap.start".into(), message: format!("{e:?}") })?;

        // Keep the device + tap + ctx alive for the capture lifetime. stop() drops them.
        // (Leak intentionally here; the process lives for the session.)
        std::mem::forget(started);
        std::mem::forget(proc_id);
        std::mem::forget(agg);
        std::mem::forget(tap);
        // ctx is leaked via into_raw; reclaim in stop via a stored pointer if needed.
        // For Phase 2a the tap runs for the session; Drop-equivalent cleanup is a follow-up.
        let _ = ctx_ptr;
        Ok(())
    }

    fn stop(&mut self) {
        // Phase 2a: capture ends with the process. Full teardown (remove IO proc,
        // destroy aggregate, reclaim ctx) is a follow-up; recording is already
        // flushed by the recorder's close(). Flagging as a known gap.
    }
}
```

> **Implementer note:** the `cidre` callback signature and `cat::AudioBufList`/`AudioTimeStamp` paths above mirror meetily's working `core_audio.rs`. If `cidre` at rev `a9587fa` names something differently (e.g., `create_io_proc_id` arg layout, `cf::ArrayOf`), align to meetily's file exactly — it compiles and runs. The **teardown leak** (`std::mem::forget`) is acceptable for Phase 2a's session-scoped tap; record a follow-up task for proper aggregate/IOProc teardown.

- [ ] **Step 4: Build (macOS) + clippy**

Run: `cargo clippy -p asr-memo-core --all-targets -- -D warnings && cargo test -p asr-memo-core audio::system`
Expected: clippy clean; `0 passed; 1 ignored`.

- [ ] **Step 5: Commit**

```bash
git add crates/asr-memo-core/src/audio/system.rs crates/asr-memo-core/src/audio/system
git commit -m "feat(v2-2a): macOS Core Audio global tap (cidre) — the system-audio fix"
```

---

### Task 8: App integration — capture mode commands + `audio-health` event

**Files:**
- Modify: `crates/asr-memo-app/src/state.rs` (add capture state)
- Modify: `crates/asr-memo-app/src/commands.rs` (add `start_capture`, `stop_capture`, `reveal_recording` + `audio-health` payload)
- Modify: `crates/asr-memo-app/src/lib.rs` (register the 3 commands)

**Interfaces:**
- Consumes: `asr_memo_core::audio::{mixer::MixedCapture, mic::CpalMicrophoneCapture, record::WavWriter, health::{HealthSnapshot, HealthFlag}}`, `asr_memo_core::traits::AudioCapture`, `AppState`, `AppHandle`, `Emitter`.
- Produces: three commands returning `serde_json::Value`; a new `audio_health` event payload `{type:"audio_health", mic:{rms,peak}, system:{rms,peak}, flags:[...]}`; `AppState.capture` holds the running `MixedCapture` + `WavWriter`.

- [ ] **Step 1: Extend `AppState`**

`crates/asr-memo-app/src/state.rs`:

```rust
//! Shared app state: the running session, transcript, and capture session.

use std::sync::Mutex;

use asr_memo_core::audio::mixer::MixedCapture;
use asr_memo_core::audio::record::WavWriter;
use asr_memo_core::session::Session;
use asr_memo_core::types::SegmentDto;

#[derive(Default)]
pub struct AppState {
    pub session: Mutex<Option<Session>>,
    pub transcript: Mutex<Vec<SegmentDto>>,
    pub capture: Mutex<Option<CaptureSession>>,
}

pub struct CaptureSession {
    pub mixer: MixedCapture,
    pub writer: WavWriter,
    pub path: String,
}
```

- [ ] **Step 2: Add the `audio-health` payload helper + the 3 commands**

In `crates/asr-memo-app/src/commands.rs`, add a pure helper (unit-tested) and the commands:

```rust
use asr_memo_core::audio::health::{HealthFlag, HealthSnapshot};

/// audio_health event payload — locked shape consumed by app.js.
pub fn audio_health_payload(snap: &HealthSnapshot) -> serde_json::Value {
    let flags: Vec<&str> = snap.flags.iter().map(|f| match f {
        HealthFlag::SystemSilent => "system_silent",
        HealthFlag::RateChanged => "rate_changed",
    }).collect();
    serde_json::json!({
        "type": "audio_health",
        "mic": { "rms": snap.mic.rms, "peak": snap.mic.peak },
        "system": { "rms": snap.system.rms, "peak": snap.system.peak },
        "flags": flags,
    })
}

#[tauri::command(rename_all = "snake_case")]
pub fn start_capture<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AppState>,
    sources: Vec<String>,
    meeting_name: Option<String>,
) -> serde_json::Value {
    use asr_memo_core::audio::mic::CpalMicrophoneCapture;
    use asr_memo_core::audio::mixer::MixedCapture;
    use state::CaptureSession;

    if state.capture.lock().unwrap().is_some() {
        return err("capture.busy", "a capture is already running", "Stop the current recording first.");
    }
    let want_mic = sources.iter().any(|s| s == "microphone");
    let want_sys = sources.iter().any(|s| s == "system");
    if !want_mic && !want_sys {
        return err("capture.sources", "no sources selected", "Choose microphone and/or system audio.");
    }

    let dir = recording_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return err("capture.dir", &e.to_string(), "Could not create the recordings directory.");
    }
    let name = meeting_name.unwrap_or_else(|| "meeting".into());
    let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let path = dir.join(format!("{name}-{stamp}.wav"));

    let mut writer = match WavWriter::create(&path, 48000) {
        Ok(w) => w,
        Err(e) => return err("capture.file", &e.to_string(), "Could not open the recording file."),
    };

    let app_for_health = app.clone();
    let writer_for_cb = std::sync::Arc::new(std::sync::Mutex::new(writer));
    let writer_for_record = writer_for_cb.clone();
    let on_health = Box::new(move |snap: HealthSnapshot| {
        // Record the mixed stream is handled via the frame callback below;
        // here we just emit health.
        let _ = app_for_health.emit("backend-event", audio_health_payload(&snap));
    });

    let mic: Option<Box<dyn asr_memo_core::traits::AudioCapture>> =
        if want_mic { Some(Box::new(CpalMicrophoneCapture::new())) } else { None };
    let sys: Option<Box<dyn asr_memo_core::traits::AudioCapture>> =
        if want_sys && cfg!(target_os = "macos") {
            Some(Box::new(asr_memo_core::audio::system::macos::CoreAudioTapCapture::new()))
        } else if want_sys {
            None // Phase 2b: Windows/Linux system capture not implemented.
        } else { None };

    let app_for_frames = app.clone();
    let wr = writer_for_record.clone();
    let mut mixer = MixedCapture::new(mic, sys, on_health);
    if let Err(e) = mixer.start(Box::new(move |frame| {
        // Upsample 16k→48k for the recording (simple linear) and write.
        if let Ok(mut w) = wr.lock() {
            let up = upsample_16_to_48(&frame.pcm);
            let _ = w.write_samples(&up);
        }
        let _ = app_for_frames.emit("backend-event", serde_json::json!({
            "type": "capture_frame", "samples": frame.pcm.len(), "t": frame.t_end
        }));
    })) {
        return err("capture.start", &e.to_string(), "Could not start audio capture.");
    }

    // Reclaim the writer from the Arc for state (it stays shared via the Arc clone).
    let path_str = path.to_string_lossy().into_owned();
    let _ = writer_for_cb; // keep Arc alive via the closure clones
    *state.capture.lock().unwrap() = Some(CaptureSession {
        mixer,
        writer: take_writer(writer_for_record),
        path: path_str.clone(),
    });
    let _ = app.emit("backend-event", serde_json::json!({"type":"status","status":"recording"}));
    serde_json::json!({"path": path_str})
}

fn upsample_16_to_48(input: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len() * 3);
    for &s in input { out.extend_from_slice(&[s, s, s]); } // zero-order hold ×3
    out
}

#[tauri::command(rename_all = "snake_case")]
pub fn stop_capture(state: State<'_, AppState>) -> serde_json::Value {
    let session = state.capture.lock().unwrap().take();
    match session {
        Some(mut s) => {
            s.mixer.stop();
            if let Err(e) = s.writer.close() {
                return err("capture.close", &e.to_string(), "Recording may be incomplete — check the file.");
            }
            serde_json::json!({"path": s.path})
        }
        None => serde_json::json!({"path": serde_json::Value::Null}),
    }
}

#[tauri::command(rename_all = "snake_case")]
pub fn reveal_recording(path: String) -> serde_json::Value {
    #[cfg(target_os = "macos")]
    {
        let p = std::path::Path::new(&path);
        if p.exists() {
            let _ = std::process::Command::new("open").arg("-R").arg(p).spawn();
            return serde_json::json!({"revealed": true});
        }
    }
    err("capture.reveal", "file not found", "Record something first, then reveal it.")
}

fn recording_dir() -> std::path::PathBuf {
    if let Some(proj) = dirs_next() {
        return proj.join("recordings");
    }
    std::path::PathBuf::from("recordings")
}

fn dirs_next() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(|h| {
        std::path::PathBuf::from(h).join("Library/Application Support/ASR-Memo")
    })
}
```

> **Implementer note:** the `writer`-shared-via-`Arc<Mutex>` dance is because `start_capture` must keep writing frames from the mixer callback AND hand a `WavWriter` to `AppState` for `close()` on stop. The clean version: store `Arc<Mutex<WavWriter>>` in `CaptureSession` (not `WavWriter`), and `stop_capture` locks it to `close()`. Apply that — drop the `take_writer` placeholder and make `CaptureSession.writer: Arc<Mutex<WavWriter>>`. Add `dirs` crate to `asr-memo-app/Cargo.toml` deps, and `chrono` (already? no — add `chrono = "0.4"`). Update the crate Cargo.toml accordingly.

Add to `commands.rs` test module:

```rust
    #[test]
    fn audio_health_payload_shape() {
        let snap = asr_memo_core::audio::health::HealthSnapshot {
            mic: asr_memo_core::audio::health::SourceHealth { rms: 0.1, peak: 0.5 },
            system: asr_memo_core::audio::health::SourceHealth { rms: 0.0, peak: 0.0 },
            flags: vec![asr_memo_core::audio::health::HealthFlag::SystemSilent],
        };
        let v = audio_health_payload(&snap);
        assert_eq!(v["type"], "audio_health");
        assert_eq!(v["mic"]["peak"], 0.5);
        assert_eq!(v["system"]["rms"], 0.0);
        assert_eq!(v["flags"][0], "system_silent");
    }
```

- [ ] **Step 3: Register the commands**

`crates/asr-memo-app/src/lib.rs` `invoke_handler!` gains `commands::start_capture, commands::stop_capture, commands::reveal_recording`.

- [ ] **Step 4: Add deps + run gate**

Add to `crates/asr-memo-app/Cargo.toml`: `chrono = "0.4"`, `dirs = "5"`. Then:

Run: `source "$HOME/.cargo/env" && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: all green (incl. the new `audio_health_payload_shape` test). Resolve the `writer`-as-`Arc<Mutex<>>` simplification from the implementer note before this step.

- [ ] **Step 5: Commit**

```bash
git add crates/asr-memo-app
git commit -m "feat(v2-2a): capture mode (start/stop/reveal) + audio-health event"
```

---

### Task 9: UI — Record mode, level meters, warning banner

**Files:**
- Modify: `crates/asr-memo-app/ui/index.html`
- Modify: `crates/asr-memo-app/ui/app.js`
- Modify: `crates/asr-memo-app/ui/styles.css`

**Interfaces:**
- Consumes: the new commands + `audio_health`/`capture_frame` events from Task 8.
- Produces: a Record button + source checkboxes reused on the Ready screen, two level meters (mic/system), and a warning banner that appears on `system_silent`/`rate_changed`.

- [ ] **Step 1: Add UI elements to `ui/index.html`**

On the Ready screen (after the existing `.actions` with Start live/Import), add a Record affordance + meters + banner:

```html
      <div class="actions">
        <button id="startBtn" class="btn btn--primary">Start live</button>
        <button id="importBtn" class="btn btn--ghost">Import audio…</button>
      </div>
      <hr class="rule" />
      <h3>Capture &amp; record (audio only)</h3>
      <p class="lede">Records mic + system audio to a WAV — the live transcript returns with ASR (Phase 3).</p>
      <div class="meters">
        <div class="meter"><span>mic</span><div class="meter__bar"><div id="micLevel" class="meter__fill"></div></div></div>
        <div class="meter"><span>system</span><div class="meter__bar"><div id="sysLevel" class="meter__fill meter__fill--sys"></div></div></div>
      </div>
      <div class="actions">
        <button id="recordBtn" class="btn btn--ghost">Record</button>
        <span id="recordStatus" class="chip"></span>
      </div>
      <div id="audioBanner" class="banner banner--warn hidden"></div>
```

- [ ] **Step 2: Wire it in `ui/app.js`**

Add (and extend the `onBackendEvent` switch with `audio_health`/`capture_frame`):

```javascript
// ---- capture / record (Phase 2a) ----
async function startCapture() {
  const sources = [];
  if ($("srcMic")?.checked) sources.push("microphone");
  if ($("srcSys")?.checked) sources.push("system");
  const res = await api().start_capture(sources, null);
  if (res?.error) { showError(res.error); return; }
  $("recordStatus").textContent = "recording…";
  $("recordBtn").textContent = "Stop & reveal";
}
async function stopCapture() {
  const res = await api().stop_capture();
  $("recordStatus").textContent = "";
  $("recordBtn").textContent = "Record";
  if (res?.path) await api().reveal_recording(res.path);
  else if (res?.error) showError(res.error);
}

function setMeter(elId, rms) {
  const pct = Math.min(100, Math.max(0, Math.log10(1 + rms * 9) * 100));
  const el = $(elId); if (el) el.style.width = pct.toFixed(1) + "%";
}

// in onBackendEvent switch, add:
      case "audio_health": {
        setMeter("micLevel", evt.mic?.rms || 0);
        setMeter("sysLevel", evt.system?.rms || 0);
        const banner = $("audioBanner");
        const flags = evt.flags || [];
        if (flags.includes("system_silent")) {
          banner.textContent = "⚠️ System audio is silent — check the Audio capture permission (macOS 14.4+).";
          banner.classList.remove("hidden");
        } else if (flags.includes("rate_changed")) {
          banner.textContent = "Audio device changed (e.g. Bluetooth) — capture reconfigured; check levels.";
          banner.classList.remove("hidden");
        } else {
          banner.classList.add("hidden");
        }
        break;
      }

// in the bootstrap wiring (next to startBtn/importBtn onclicks):
  const rb = $("recordBtn");
  if (rb) rb.onclick = () => (rb.textContent === "Record" ? startCapture() : stopCapture());
```

- [ ] **Step 3: Add CSS to `ui/styles.css`**

```css
.rule { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: 1rem 0; }
.meters { display: flex; flex-direction: column; gap: .5rem; margin: .5rem 0; }
.meter { display: flex; align-items: center; gap: .5rem; }
.meter > span { width: 4rem; color: var(--muted, #8a93a6); font-size: .8rem; }
.meter__bar { flex: 1; height: 8px; background: rgba(255,255,255,.08); border-radius: 4px; overflow: hidden; }
.meter__fill { height: 100%; width: 0%; background: #1A7F64; transition: width .08s linear; }
.meter__fill--sys { background: #2D7FF9; }
.banner { padding: .6rem .8rem; border-radius: 8px; font-size: .9rem; margin-top: .5rem; }
.banner--warn { background: rgba(228,77,80,.16); border: 1px solid rgba(228,77,80,.4); color: #f0a0a2; }
```

- [ ] **Step 4: Build + smoke-launch**

Run: `source "$HOME/.cargo/env" && cargo build -p asr-memo-app && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: build clean; gate green.

- [ ] **Step 5: Commit**

```bash
git add crates/asr-memo-app/ui
git commit -m "feat(v2-2a): Record UI — meters + system-silent/rate warning banner"
```

---

### Task 10: macOS packaging (permission strings) + 3-OS CI gate

**Files:**
- Modify: `crates/asr-memo-app/Info.plist` (create if absent; add usage strings)
- Modify: `crates/asr-memo-app/tauri.conf.json` (reference the plist / macos config)
- Manual: run the real-call gate

**Interfaces:**
- Consumes: the engine from Tasks 2–9.
- Produces: a macOS bundle that prompts for mic + audio-capture permission; the 2a gate verified.

- [ ] **Step 1: Add macOS usage strings**

Create/extend `crates/asr-memo-app/Info.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>NSMicrophoneUsageDescription</key>
  <string>ASR-Memo captures your microphone to transcribe meetings locally on your device.</string>
  <key>NSAudioCaptureUsageDescription</key>
  <string>ASR-Memo captures system audio (remote participants) to transcribe meetings locally on your device.</string>
</dict>
</plist>
```

Reference it in `tauri.conf.json` under `bundle` → `macOS` (Tauri 2 merges this plist):

```json
    "macOS": {
      "infoPlist": "Info.plist"
    }
```

- [ ] **Step 2: Verify the full 3-OS CI gate locally**

Run: `source "$HOME/.cargo/env" && cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: all green on macOS. Push to run Windows + Linux CI.

- [ ] **Step 3: Commit + push (CI runs on all 3 OSes)**

```bash
git add crates/asr-memo-app/Info.plist crates/asr-memo-app/tauri.conf.json
git commit -m "feat(v2-2a): macOS mic+audio-capture permission strings; 3-OS CI gate"
git push -u origin feat/v2-phase2a-audio-engine
```

- [ ] **Step 4: Manual 2a gate (on a macOS 14.4+ machine)**

1. `cargo run -p asr-memo-app`, grant mic + audio-capture prompts.
2. Start a Teams/Zoom/Meet call; click **Record** with mic + system checked.
3. After ~60 s click **Stop & reveal**; play the WAV → **both sides audible**.
4. Mid-call, revoke System Settings → Privacy → Audio Capture; within a few seconds the **"system audio is silent"** banner appears.
5. Switch a Bluetooth headset profile mid-call → `rate_changed` banner + recording continues.
6. Run the `#[ignore]` hardware tests: `cargo test -p asr-memo-core -- --ignored audio::mic audio::system`.

Expected: all pass → Phase 2a gate met.

---

## Self-Review (done at write time)

**Spec coverage:** cpal mic (Task 6) + cidre global tap tap-only aggregate (Task 7) ✓; 50 ms align + RMS dynamic mix 10–90% + soft-scale (Task 5) ✓; persistent rubato resampler + rate-change reconfigure (Task 2) ✓; runtime Bluetooth rate tracking (Task 7 io_proc rate check + Task 2 set_rates) ✓; incremental 48 kHz WAV record (Task 4) ✓; audio-health events + minimal UI levels + warning banner (Tasks 3, 8, 9) ✓; offline tests + `#[ignore]` hardware tests (Tasks 2–5 offline; 6–7 ignored) ✓; macOS permission strings (Task 10) ✓; 3-OS CI (Task 1 libasound2-dev + Task 10) ✓; VAD deferred to Phase 3 (no VAD task — intentional) ✓; 2b deferred ✓.

**Known gaps flagged for the implementer (not placeholders):** (a) Task 5 `on_health` field should be `Option<Box<…>>` (take in `start`); (b) Task 7 cidre teardown is `std::mem::forget` (session-scoped; follow-up); (c) Task 8 `CaptureSession.writer` should be `Arc<Mutex<WavWriter>>`. Each is a concrete simplification with the resolution stated in-line.

**Type consistency:** `AudioFrame`/`AudioSource`/`CoreError` (Phase-1 traits) used unchanged; `PersistentResampler`, `HealthMonitor`/`HealthSnapshot`/`HealthFlag`, `WavWriter`, `MixedCapture`, `CpalMicrophoneCapture`, `CoreAudioTapCapture`, `CaptureSession` names consistent across their defining task and consumers.
