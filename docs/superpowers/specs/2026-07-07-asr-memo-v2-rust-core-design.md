# ASR-Memo v2 — Tauri App with Full Rust Core (Design)

> **Status:** Approved by maintainer 2026-07-07.
> **Supersedes:** the architecture direction of `docs/plans/cross-platform-nemo.md`
> (NeMo/PyTorch backends inside the Python app). That plan is retired; the
> implementation plan derived from this spec replaces it.
> **Informed by:** analysis of [Zackriya-Solutions/meetily](https://github.com/Zackriya-Solutions/meetily)
> (MIT), a trending local-first meeting transcription app (Tauri + Rust core,
> whisper.cpp / Parakeet-ONNX-CPU, no diarization in the open-source edition).

## 1. Context and motivation

ASR-Memo today is a macOS-only Python app: `pywebview` UI over the in-process
`meeting_asr` library (streaming Nemotron 3.5 ASR via ONNX Runtime CoreML EP ∥
Sortformer diarization via CoreML, fused by word timestamps), with a Swift
Core Audio Process-Tap helper for system audio.

Two problems triggered this redesign:

1. **System-audio reliability.** In real Teams calls the remote party is
   sometimes not transcribed at all — only the local mic. Meetily's code
   documents the exact failure modes: the macOS process tap **returns silence
   (all zeros) when the audio-capture permission is missing** (no error), and
   **Bluetooth headsets renegotiate sample rates mid-call** (A2DP↔HFP) which
   breaks captures that read the device rate once at startup (ours does).
   Our plain summing mixer also lets a quiet remote caller be drowned by the
   local mic.
2. **Perceived accuracy degradation.** Meetily protects accuracy with a funnel
   in front of the model (Silero VAD gating, RMS/peak silence rejection to
   prevent hallucinations, one persistent resampler per stream after finding
   that per-chunk resampler recreation amplified energy ~173%). We have none of
   these guards; our degradation is at least as plausibly pipeline hygiene as
   the FP16 ONNX export itself.

Meetily validates two strategic facts for us:

- A **Tauri + Rust core** ships one robust cross-platform binary (macOS,
  Windows, Linux) with native capture on each OS — no Python runtime, no
  helper processes. They archived their Python/FastAPI backend after moving
  everything into the Rust core.
- **ONNX Runtime is sufficient for NVIDIA RNNT models cross-platform** —
  their Parakeet-TDT 0.6B runs live meetings on the **CPU EP alone** (with
  optional int8 quantization). A NeMo/PyTorch dependency stack is unnecessary.

What meetily does **not** have (open-source edition): speaker diarization
(paid-roadmap item), true streaming ASR (their "live" transcript is
VAD-segmented batch), word-level timestamps, or timestamp fusion. Those are
our differentiators and this design preserves all of them.

## 2. Decisions (confirmed with maintainer, 2026-07-07)

- **D1 — Architecture:** Rewrite the shell as **Tauri 2.x with a full Rust
  core** (option B over: staying Python, or a Python-sidecar hybrid).
- **D2 — Scheduling:** Audio-engine work and inference-port work run as **two
  parallel tracks**; the macOS Teams bug is fixed *by* the new Rust capture
  layer rather than patched first in Python.
- **D3 — Migration:** **Full Rust core** (option over "Python sidecar first"):
  ASR, diarization, and fusion are ported to Rust (`ort`). Python remains only
  as the dev-time **reference implementation and validation harness**.
- **D4 — Design approved as presented** (UI reuse, SQLite, same-repo
  workspace, parity gates, risk plan below).

## 3. Goals / non-goals

**Goals**

- One cross-platform desktop app (macOS Apple Silicon, Windows x64,
  Linux x64) installed as a single Tauri bundle per OS.
- Meetily-grade audio robustness: per-stream health monitoring, runtime
  sample-rate tracking, dynamic mixing, VAD gating — on all three OSes.
- Keep the model stack: **Nemotron 3.5 ASR Streaming Multilingual 0.6B**
  (3-graph cache-aware streaming RNNT, ONNX FP16) and **Streaming Sortformer
  4spk-v2.1**, now both via ONNX Runtime (`ort`) with per-platform execution
  providers.
- Keep the product differentiators: live **diarized** transcript, word-level
  timestamps, diarization∥ASR timestamp fusion, ~40-language multilingual ASR.
- Local-first storage (SQLite) and Markdown/JSON export.

**Non-goals (this spec)**

- LLM meeting summaries (future phase; meetily's Ollama/provider pattern is
  the reference when we get there).
- Changing models or training anything.
- Mobile, meeting auto-join, calendar integration.
- Maintaining the Python app as a product after Rust parity (it is archived,
  meetily-style, once the parity gate passes).

## 4. Architecture

```
repo (Cargo workspace added at root)
├── crates/
│   ├── asr-memo-core/        # library: audio engine, inference, fusion, session
│   └── asr-memo-app/         # Tauri 2.x app (commands/events, tray, packaging)
│       └── ui/               # existing static HTML/CSS/JS from app/web, ported
├── src/meeting_asr/          # Python REFERENCE implementation (frozen, dev-only)
├── validation/               # Python WER/DER/lang-ID harness — the parity gate
└── docs/plans/…              # this spec's implementation plan supersedes
                              # cross-platform-nemo.md
```

- **UI:** reuse our static HTML/CSS/JS (no Node build step, no Next.js).
  `pywebview js_api` calls become Tauri `invoke` commands; the emit-pump
  becomes Tauri events (`transcript-segment`, `audio-health`, `readiness`,
  `session-state`). The existing DTO shapes in `app/dto.py` are carried over
  as `serde` structs so the UI port is mechanical.
- **Swift AudioTap helper: retired.** System audio on macOS moves in-process
  via `cidre` (Rust bindings to Core Audio), same CATapDescription mechanism.
- **Python:** `src/meeting_asr` is frozen as the reference implementation used
  to generate golden fixtures; `validation/` remains the accuracy gate. No
  Python ships in the app bundle.

## 5. Rust core components

### 5.1 Audio engine (Track A)

- **Microphone:** `cpal` on all OSes (F32/I16/I32/I8 input paths, device
  enumeration per platform module like meetily's `devices/platform/*`).
- **System audio:**
  - *macOS:* `cidre` Core Audio **global mono process tap** wrapped in a
    **tap-only private aggregate device** (meetily's echo fix: never tap +
    output device together). Permission failure is detected by level
    monitoring (tap yields zeros when denied) and surfaced in the UI.
  - *Windows:* **WASAPI loopback** via `cpal` (open an output device as an
    input stream; enumerate via the WASAPI host with default-host fallback).
  - *Linux:* PulseAudio/PipeWire **monitor source** via `cpal`.
- **Pipeline** (port of meetily's proven recipe):
  - Ring-buffer alignment of mic/system in **50 ms windows**.
  - **RMS-based dynamic mixing** (ratios clamped 10–90%) with **soft-scaling**
    instead of hard clipping; optional RNNoise (`nnnoiseless`) denoise stage.
  - **One persistent `rubato` resampler per stream** (fixed 512-sample chunks
    with an accumulation buffer for variable input sizes).
  - **Runtime sample-rate tracking** read inside the capture callback
    (Bluetooth A2DP↔HFP renegotiation, device switches).
  - **Stream-health monitoring:** per-source RMS/peak metrics emitted to the
    UI; sustained silence on an active SYSTEM stream raises a visible warning
    ("system audio appears silent — check Screen/Audio capture permission").
- **VAD gate:** Silero VAD (`silero-rs`, 16 kHz) + RMS/peak floor in front of
  the ASR. Prevents hallucination on silence/noise and cuts inference load.
  Note: the VAD gates *inference*, not the diarizer clock — frame timestamps
  stay continuous so fusion is unaffected.
- **Recording path:** in parallel with transcription, the mixed stream is
  saved (48 kHz source fidelity) for playback/re-transcription, with
  incremental saving so a crash never loses a meeting.

### 5.2 Inference (Track B)

- **ASR — `NemotronOrtTranscriber`:** direct port of
  `src/meeting_asr/asr/nemotron_onnx.py` (3 sessions: encoder / decoder /
  joint; cache-aware streaming step; word-level tokens with `t_start`/`t_end`;
  `NEMOTRON_LANGUAGES` + language-hint prompt logic) onto `ort`.
- **Diarization — `SortformerOrtDiarizer`:** port of the streaming step,
  arrival-order speaker labeling, and `ACTIVATION_THRESHOLD` decode from
  `sortformer_coreml.py`, running a **Sortformer ONNX export** on `ort`.
  **Spike S1 (blocking for this component only):** confirm a usable streaming
  Sortformer ONNX export — check FluidInference's Hugging Face repos first,
  else export from NVIDIA's `.nemo` checkpoint ourselves (one-time, dev-side).
- **Execution providers (runtime-selected, replacing `backends/device.py`):**
  - macOS: CoreML EP → CPU fallback.
  - Windows: CUDA EP → DirectML EP → CPU.
  - Linux: CUDA EP → CPU.
  - CPU path offers int8-quantized model variants (meetily-validated approach
    for RNNT-on-CPU). EP failures degrade **per graph** to CPU rather than
    failing the session.
- **Fusion:** direct port of `fusion/aligner.py` (pure logic).
- **Session orchestrator:** port of `session.py`/`pipeline.py` semantics —
  live segment events, diarizer∥ASR concurrency, finalization, export.

### 5.3 App services

- **Model manager:** `hf-hub` downloads the same pinned repos into the same
  cache (`~/.cache/meeting_asr/models`); readiness checks (models present,
  permissions, EP availability) reported to the UI as today.
- **Storage:** SQLite via `sqlx` — meetings, transcript segments (speaker,
  language, timestamps), session metadata. Markdown/JSON export ported from
  `export/`.
- **Packaging:** Tauri bundler (`.dmg`/`.msi`/`.AppImage`+`.deb`), build-time
  GPU feature flags with a meetily-style auto-detect build script; CI matrix
  (3 OSes) runs the offline suite only.

## 6. Data flow

```
mic (cpal) ──┐                                            ┌─▶ recorder (48 kHz, incremental save)
             ├─ align 50 ms ─ dynamic mix ─ soft-scale ───┤
system tap ──┘   (health metrics ──▶ UI events)           └─▶ resample 16 kHz mono
                                                               ├─▶ Silero VAD ─ gate ─▶ Nemotron RNNT (ort) ─ words+timestamps ─┐
                                                               └────────────────────▶ Sortformer (ort) ─ 80 ms speaker frames ──┤
                                                                                                          fusion aligner ◀──────┘
                                                                                                               │
                                                                            SQLite + Tauri event ─▶ UI live transcript / export
```

## 7. Parallel tracks and parity gates

- **Track A (audio engine)** is independently shippable: a record-to-file mode
  exercises capture/mix/health on all OSes before inference lands. It is also
  the fix for the Teams bug (permission-silence detection + rate tracking +
  dynamic mixing).
- **Track B (inference)** proceeds ASR → diarizer (behind spike S1) → fusion.
- **Gates:**
  1. *Golden-fixture parity:* Rust ports must reproduce Python reference
     outputs on recorded fixture streams (tokens, timestamps, speaker frames)
     within defined tolerances.
  2. *Accuracy parity:* `validation/` WER / DER / language-ID on the labeled
     clip set — Rust app vs current macOS Python baseline (also settles how
     much of the accuracy complaint was pipeline vs model).
  3. Python app archived only after both gates pass on macOS and the app runs
     on Windows + one Linux distro.

## 8. Testing

Adopt the repo's offline discipline in Rust: trait-based fakes for capture /
ASR / diarizer (mirroring `tests/_fakes.py`), deterministic fixture tests, no
network or models in CI. `needs_models` / `needs_hardware`-style gating via
cargo features or `#[ignore]` + explicit CI jobs. The Python suite keeps
guarding the reference implementation until archive.

## 9. Risks (ordered)

1. **Sortformer ONNX export (S1).** May not exist ready-made; self-export from
   `.nemo` needs a one-time dev-side NeMo/torch environment. *Fallback:* ship
   ASR-only on Windows/Linux first (still ≥ meetily), diarization follows.
2. **RNNT streaming decode subtleties in Rust.** Cache tensors, blank-token
   loops, token→word merge. *Mitigation:* golden fixtures from the Python
   reference at every step boundary.
3. **`ort` EP coverage** (CoreML/DirectML op support for these graphs).
   *Mitigation:* per-graph CPU fallback; CPU-int8 is a validated floor.
4. **Linux capture fragmentation** (Pulse vs PipeWire vs ALSA). *Mitigation:*
   monitor-source approach + explicit device picker; Linux is build-from-source
   tier at first (as meetily does).
5. **Rewrite scope.** *Mitigation:* two tracks, UI carried over, DTOs carried
   over, Python kept runnable until gates pass.

## 10. Success criteria

- A Teams/Zoom/Meet call on macOS produces a transcript containing **both**
  sides, attributed to different speakers; pulling the system-audio permission
  mid-session raises a visible health warning instead of silent mic-only text.
- Same app binary pattern installs and runs live diarized transcription on
  Windows (GPU and CPU-only) and Linux.
- WER/DER on the validation set ≤ current macOS Python baseline.
- Live word latency no worse than today's macOS app; no regression from
  VAD gating on continuous speech.
- Offline test suite green on the 3-OS CI matrix with no model downloads.
