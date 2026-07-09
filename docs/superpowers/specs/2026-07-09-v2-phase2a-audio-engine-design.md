# ASR-Memo v2 — Phase 2a: macOS Audio Engine (Design)

> **Status:** Approved by maintainer 2026-07-09.
> **Parent spec:** `docs/superpowers/specs/2026-07-07-asr-memo-v2-rust-core-design.md`
> (the v2 master design). This document refines that spec's Phase 2 into a
> concrete Phase 2a (macOS-first) and Phase 2b (Windows/Linux port, specced later).
> **Refines the parent's phase split:** Silero VAD moves from Phase 2 to Phase 3
> (it gates inference, which arrives in Phase 3; it has no consumer in 2a).

## 1. Goal and motivation

Phase 2a delivers the real audio engine on macOS and **directly fixes the
maintainer's original bug**: in a Teams/Zoom/Meet call, the remote party was
sometimes not transcribed at all — only the local mic. Analysis of
[Zackriya-Solutions/meetily](https://github.com/Zackriya-Solutions/meetily)
(MIT) identified the causes: the macOS process tap returns **silent zeros when
the audio-capture permission is missing** (no error), **Bluetooth headsets
renegotiate sample rates mid-call** (A2DP↔HFP) which breaks captures that read
the rate once at startup, and a naive summing mixer lets a quiet remote caller
be drowned by the local mic.

**Verifiable end-state (the 2a gate):** record a real call on macOS and play
back a file with **both sides audible**; pull the system-audio permission
mid-call and see a **"system audio silent"** warning instead of silent
mic-only capture; offline suite green on the 3-OS CI.

## 2. Scope decisions (confirmed 2026-07-09)

- **D1 — Separate capture/record mode.** ASR is Phase 3; feeding real audio
  into the Phase-1 `FakeTranscriber` would produce nonsense text at real
  timestamps. So 2a adds a distinct **capture mode** (`start_capture` /
  `stop_capture` commands) that runs real mic+system capture → mix → record +
  health, with **no transcript pane**. The Phase-1 `start_live` (fake
  transcript demo) stays; real `start_live` returns in Phase 3 with actual ASR.
- **D2 — Defer Silero VAD to Phase 3.** VAD's only job is to gate inference.
  With no ASR in 2a it has no consumer, and recording must be continuous (the
  whole meeting is wanted). VAD + ASR land together in Phase 3. This adjusts
  the parent spec's Phase 2/3 split.
- **D3 — Record format/location.** WAV, 48 kHz, mixed mono, written
  incrementally to `~/Library/Application Support/ASR-Memo/recordings/`
  (`<meeting>-<timestamp>.wav`), with a "reveal in Finder" affordance. WAV is
  lossless and needs no codec dependency; compressed formats are a Phase 5
  packaging option.
- **D4 — macOS-first.** Phase 2a is macOS only (the maintainer's pain).
  Windows (WASAPI loopback) + Linux (PipeWire monitor) are Phase 2b, reusing
  this phase's `mixer`/`resample`/`health`/`record` unchanged behind the
  `AudioCapture` trait.

## 3. Non-goals (Phase 2a)

- Live transcript / ASR / diarization (Phase 3 / 4).
- Silero VAD and speech-region sidecars (Phase 3).
- Windows and Linux system-audio capture (Phase 2b).
- RNNoise denoise, compressed recording, SQLite persistence (later phases).
- Changing the Phase-1 `start_live` contract (it remains the fake demo).

## 4. Architecture

New subsystem under `crates/asr-memo-core/src/audio/`, plugging into the
Phase-1 `AudioCapture` trait (`traits.rs`) so the existing `Session` seam is
unchanged. Each module has one responsibility and is independently testable.

```text
mic (cpal, 48k) ─┐                              ┌─▶ record.rs (48 kHz WAV, incremental flush)
                 ├─ 50 ms align ─ RMS dynamic  ─┤
system tap (cidre,─┘  mix + soft-scale          └─▶ resample 16 kHz ─▶ (Phase 3: VAD + ASR)
  48k, mono global)   │
   ↑                  └─ health.rs ──▶ audio-health event ──▶ UI (levels + "system silent" warning)
   └ rate.rs (runtime sample-rate tracking; Bluetooth renegotiation)
```

### 4.1 Modules

- **`mic.rs` → `CpalMicrophoneCapture: AudioCapture`** — cpal input stream,
  emits frames resampled to 16 kHz mono (the resampler lives in `resample.rs`,
  reused). Device selection via cpal's default input device (configurable later).
- **`system/macos.rs` → `CoreAudioTapCapture: AudioCapture`** (`cfg(target_os =
  "macos")`) — `cidre` Core Audio **global mono process tap** wrapped in a
  **tap-only private aggregate device** (never tap + output device together —
  that caused meetily's echo). Emits the tap stream resampled to 16 kHz mono.
  Non-macOS targets get a stub that returns `CoreError::Capture`.
- **`mixer.rs` → `MixedCapture: AudioCapture`** — owns a mic + a system
  capture; aligns them in **50 ms windows** (ring buffer, `VecDeque`), applies
  **RMS-based dynamic mixing** (per-window mic/system RMS → ratios clamped to
  10–90%) with **soft-scaling** (sum > 1.0 is scaled down, never hard-clipped),
  and emits mixed 16 kHz mono `AudioFrame`s on the session clock. Port of the
  Python reference's `MultiSourceMixer`/`CompositeCapture` logic, augmented with
  meetily's RMS dynamic mixing.
- **`resample.rs`** — wrapper around a **persistent `rubato` resampler per
  stream** with a 512-sample accumulation buffer. Meets meetily's hard-won
  constraint: never recreate the resampler per chunk (that amplified energy
  ~173% and produced wrong output sizes).
- **`rate.rs`** — runtime sample-rate tracking read **inside the capture
  callback** (cpal/cidre expose the device's nominal rate). On a rate change
  (Bluetooth profile switch, device swap) the resampler is transparently
  reconfigured for the new ratio; a `rate_changed` flag is emitted on health.
- **`health.rs`** — computes per-source RMS/peak over recent windows and
  detects sustained silence on an active SYSTEM stream (the permission-denied
  signature: the tap yields all-zeros). Emits the `audio-health` event.
- **`record.rs`** — incremental WAV writer (48 kHz mixed stream): writes the
  header on open, appends PCM on each window, patches the length on
  close/flush so a crash leaves a valid (shorter) file.

### 4.2 App integration (`crates/asr-memo-app`)

- New commands: `start_capture` (sources: mic and/or system, meeting name),
  `stop_capture` (→ finalizes WAV, returns path), `reveal_recording` (Finder).
- New Tauri event `audio-health`: `{type:"audio_health", mic:{rms,peak},
  system:{rms,peak}, flags:["system_silent"|"rate_changed"]}`. (New event type
  alongside the Phase-1 `status`/`segment`/`error`.)
- UI: minimal, static — a **Record** affordance, two **level meters**
  (mic/system), and a **warning banner** that appears on `system_silent`/
  `rate_changed`. No design overhaul; reuses existing CSS idioms.

## 5. Capture reliability (the bug fix, in detail)

1. **Tap-only aggregate.** The `cidre` aggregate device includes the tap and
   the main output device's UID as the sub-device **but not a second tap of the
   same device** — the configuration that produced meetily's YouTube-audio echo.
2. **Permission-silence detection.** macOS 14.4+ returns silent zeros from the
   tap when the Audio Capture permission (`NSAudioCaptureUsageDescription`) is
   denied — there is no error callback. `health.rs` detects sustained
   all-zero SYSTEM frames on an active stream and emits `system_silent`, which
   the UI surfaces as an actionable warning ("system audio appears silent —
   check Screen/Audio capture permission"). This replaces today's failure mode
   of silently producing mic-only capture.
3. **Runtime rate tracking.** The capture callback reads the device's nominal
   sample rate each cycle (or on the device-notifications callback). On change,
   `rate.rs` reconfigures the per-stream resampler to the new ratio and emits
   `rate_changed`. Captures that read the rate once at startup (the Python
   reference's current behavior) break on Bluetooth A2DP↔HFP renegotiation;
   this fixes that.

## 6. Dependencies

Added in 2a: `cpal` (mic, all OSes), `cidre` (mac tap; git dependency,
`cfg(target_os = "macos")`), `rubato` (resampling). **Not** in 2a: `silero-rs`
(Phase 3), `nnnoiseless` (deferred). Heavy audio deps are lazy/`cfg`-gated so
the offline test suite and Windows/Linux CI compile without audio hardware.

## 7. Testing

Offline-first, mirroring Phase 1's discipline (no network/models/hardware in CI):

- **`mixer.rs`**: deterministic frame merging across misaligned mic/system
  arrival; RMS ratio computation clamped 10–90%; soft-scale under sum > 1.0.
- **`resample.rs`**: 48k→16k ratio correctness; energy preserved across
  variable-length chunks; the persistent-resampler-vs-per-chunk energy
  regression is locked by an explicit test.
- **`health.rs`**: RMS/peak over a windowed buffer; sustained-zero SYSTEM →
  `system_silent`; rate-change → `rate_changed`.
- **`record.rs`**: WAV header + data correctness on a scripted PCM stream;
  length-patch-on-close; a simulated mid-write stop yields a valid shorter file.
- **`rate.rs`**: ratio-reconfigure on a simulated rate change.
- Real `CpalMicrophoneCapture` and `CoreAudioTapCapture` are `#[ignore]`
  hardware-gated tests, run manually on a macOS 14.4+ machine.

macOS runs the real-path unit tests; Windows/Linux compile the `cfg(macos)`-
gated code via the non-mac stub so the 3-OS CI stays green.

## 8. Phase 2a gate (done when)

- A real Teams/Zoom/Meet call recorded on macOS plays back with **both sides**
  audible in the WAV.
- Pulling the system-audio permission mid-call raises the **`system_silent`**
  warning (not silent mic-only capture).
- Switching a Bluetooth headset's profile mid-call does not break capture
  (`rate_changed` fires, resampler reconfigures, recording continues).
- Offline suite green on the 3-OS CI; manual `#[ignore]` hardware tests pass.
- Requires macOS 14.4+ (process-tap availability, matching the existing Swift
  helper's requirement).

## 9. Phase 2b (deferred — specced when 2a lands)

`system/windows.rs` (WASAPI loopback via cpal: open an output device as an
input stream) and `system/linux.rs` (PulseAudio/PipeWire `.monitor` source via
cpal), both conforming to `AudioCapture` and reusing `mixer`/`resample`/
`health`/`record` unchanged. The system-audio source becomes a per-OS swap
behind the trait; nothing else in the pipeline changes.

## 10. Risks

1. **`cidre` API churn** (git dependency, `rev`-pinned). Mitigation: pin a
   specific rev; if the tap API drifts, fall back to the `coreaudio-rs` crate.
2. **macOS permission UX** (the dialog is triggered by tap creation; denial is
   silent). Mitigation: the `system_silent` detector is the primary signal;
   readiness check also reports permission state.
3. **Soft-real-time capture callback vs. allocation.** Mitigation: pre-allocate
   buffers; ring-buffer hand-off to the mixer thread; no allocation in the
   audio callback.
4. **`rubato` on the callback thread.** Mitigation: resample on the mixer
   thread, not the callback; the callback only pushes raw samples to a ring.
