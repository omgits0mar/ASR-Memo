# Phase 0 Research: Realtime Diarized Meeting Transcription Backend

**Feature**: `001-meeting-asr-backend`
**Date**: 2026-06-14
**Driving constraint (user)**: "the most low latency and optimal way."

This document resolves the open technical decisions for the lowest-latency, on-device
(macOS Apple Silicon) realization of the spec, consistent with the project constitution.

---

## Decision 1 — ASR runtime: ONNX Runtime (CoreML EP, FP16) on the M-series GPU + CPU

- **Decision**: Run NVIDIA Nemotron 3.5 ASR Streaming 0.6B as an **FP16 ONNX export**,
  executed through **ONNX Runtime with the CoreML Execution Provider** configured for
  **`MLComputeUnits = .cpuAndGPU`** (M-series Metal GPU + CPU), with a **pure CPU EP
  fallback** when CoreML is unavailable. Use the **560ms chunk** streaming configuration.
  Precision is a **config knob** (`fp16` default → `int8` → `int4` experimental); see
  Decision 8 for why FP16 is the default and the gate that governs any quantization.
- **Rationale**:
  - Architecture is a 24-layer FastConformer encoder + RNNT decoder that is *cache-aware*:
    encoder self-attention and convolution activations are cached and reused, so each
    audio frame is processed exactly once with no overlapping recomputation — this is the
    core of its low streaming latency.
  - **FP16 is the well-supported, fast path on Apple Silicon**: the M-series GPU (Metal)
    executes FP16 matmuls natively with broad op coverage, and the ONNX export removes the
    heavy NeMo/PyTorch runtime from the hot path. A 0.6B model is ~1.2 GB at FP16 — well
    within the 16 GB unified-memory envelope, so memory is *not* the binding constraint and
    does not justify lossy quantization.
  - **GPU + CPU is targeted explicitly** (`.cpuAndGPU`) rather than the ANE: the ANE is
    FP16-palettization-centric with limited FastConformer op coverage (ops fall back/copy),
    whereas the Metal GPU gives predictable FP16 throughput with full coverage and CPU picks
    up any unsupported ops. ANE (`MLComputeUnits = .all`) remains an opt-in to profile.
  - 560ms chunk is the latency/accuracy sweet spot; smaller chunks (80–320ms) are available
    if profiling shows headroom.
  - Multilingual (40 language-locales) from one checkpoint via language-ID prompt
    conditioning; `language=None` triggers auto handling — satisfies per-turn language
    (Constitution VI) and FR-007/FR-008.
- **Alternatives considered**:
  - *INT4 ONNX export (`onnx-community/...-onnx-int4`)*: **rejected as default.** For a small
    0.6B model the memory saving is non-binding, ORT's INT4 matmul (`MatMulNBits`) has poor
    CoreML-EP coverage (falls back to CPU → can *raise* latency), and weight-only INT4 risks
    measurable WER loss that is uneven across the 40 locales (low-resource/accented speech
    degrades first). Allowed only as an experiment behind the Decision 8 WER+latency gate.
  - *INT8 weight-only*: the safe middle ground (typically <1% relative WER hit, well
    supported) — the documented fallback if FP16 memory/latency ever becomes a problem.
  - *NeMo toolkit on PyTorch MPS*: works (model card confirms CPU/MPS/CUDA) and is a direct
    Metal-GPU path, but pulls in the full NeMo + PyTorch stack with higher cold-start and
    per-chunk overhead. Kept as a *reference/validation* path only.
  - *sherpa-onnx (Rust/C++)*: excellent for embedding, but adds a non-Python runtime boundary
    that complicates the in-process Python library contract. Future optimization if needed.
  - *whisper.cpp (constitution's default ASR)*: rejected — the spec mandates Nemotron, and
    Whisper is not natively streaming/cache-aware (worse live latency).

## Decision 2 — Diarization runtime: Streaming Sortformer, CoreML build

- **Decision**: Use NVIDIA **Streaming Sortformer** (`diar_streaming_sortformer_4spk-v2.1`)
  via the **CoreML conversion** (`FluidInference/diar-streaming-sortformer-coreml`),
  configured for the **low-latency profile**: chunk size 6, right context 7, FIFO 188,
  80ms frames, 16 kHz mono → ~1.04s latency, RTF ≈ 0.093 (~120× RTF on Apple Silicon).
- **Rationale**:
  - Native CoreML runs on the Apple Neural Engine / GPU — far lower latency and energy
    than PyTorch-MPS NeMo, and frees unified memory for the ASR model.
  - Sortformer is end-to-end (no separate VAD + embedding + clustering pipeline), handles
    overlapping speech, and uses an Arrival-Order Speaker Cache (AOSC) that produces
    *stable* speaker IDs across the session — directly satisfies FR-003/FR-004 and
    Constitution IV.
  - Supports up to 4 concurrent speakers (v2 4spk), matching SC-002's "up to 4 speakers."
  - 80ms frame-level output gives the temporal resolution needed to attribute ASR text.
- **Alternatives considered**:
  - *pyannote.audio 3.1* (constitution's listed diarizer): gold-standard offline DER but
    its streaming story is weaker and higher-latency than Streaming Sortformer; kept as the
    documented fallback if Sortformer's 4-speaker cap or accuracy proves limiting.
  - *Sortformer via NeMo/PyTorch-MPS*: heavier runtime; used only to validate the CoreML
    build's outputs against reference.

## Decision 3 — System-audio capture: Core Audio Process Taps via a Swift helper

- **Decision**: Capture meeting-app output with **Core Audio Process Taps**
  (`AudioHardwareCreateProcessTap` + `CATapDescription`, macOS 14.4+), implemented as a
  small **Swift helper binary** that emits raw PCM to stdout, wrapped by the Python
  `AudioCapture` module via a subprocess + pipe (the proven AudioTee pattern). Capture the
  **microphone** separately through PortAudio (`sounddevice`). Both streams are resampled
  to **16 kHz mono** and timestamped against a single monotonic session clock.
- **Rationale**:
  - Process Taps is Apple's native, lowest-latency way to capture system output per-process
    or system-wide, with explicit user permission and **no virtual audio device** (BlackHole)
    or accessibility hacks — satisfies Constitution III and FR-002/FR-015.
  - It is meeting-app-agnostic (works for Teams, Meet, Zoom, anything) — FR-002.
  - The tap API surface is Swift/C; a thin Swift helper is more reliable and lower-latency
    than driving CoreAudio through PyObjC, and isolates platform-native code behind the
    Python interface (Constitution III "behind a platform-agnostic interface").
  - Separate mic + system streams preserve a clean local-vs-remote source tag while still
    feeding one mixed 16 kHz mono stream to diarization/ASR.
- **Alternatives considered**:
  - *ScreenCaptureKit audio (macOS 13+)*: viable and also native, but Process Taps is more
    direct/lower-latency for audio-only capture and avoids screen-recording permission
    semantics. ScreenCaptureKit is the documented fallback for macOS 13.0–14.3.
  - *BlackHole / virtual aggregate device*: rejected — requires user device reconfiguration,
    adds buffering latency, and is fragile. Constitution allows it only as a last resort.
  - *PyObjC-only Process Tap*: rejected for the hot path — incomplete/awkward bindings,
    higher risk; Swift helper is cleaner.

## Decision 4 — Pipeline topology: parallel diarize + transcribe, fuse by timestamp

- **Decision**: Run Streaming Sortformer and the Nemotron streaming ASR **concurrently on
  the same mixed 16 kHz mono stream**, each maintaining its own streaming cache, then
  **fuse** the ASR token/segment timestamps against the diarization speaker timeline to
  emit `TranscriptSegment`s. A small alignment buffer (≈ the diarization latency, ~1s)
  holds ASR output until the speaker label for that time window is finalized.
- **Rationale**:
  - This is the lowest-latency, highest-accuracy topology. Cache-aware streaming RNNT
    relies on an uninterrupted audio stream to keep its encoder cache valid; **re-segmenting
    audio per speaker and feeding fixed segments into the ASR would reset/fragment that
    cache**, increasing latency and hurting accuracy — the opposite of the user's goal.
  - Diarization remains authoritative for "who spoke" and runs continuously from the start
    of capture (it is still the first interpretive stage on the captured audio); ASR provides
    "what was said." Fusion aligns the two on the shared session clock.
  - End-to-end budget: capture (~tens of ms) + max(diarization ~1.04s, ASR ~0.56s) +
    fusion buffer ≈ **1.5–2.0s turn-to-text**, within Constitution II (≤3s) and SC-001 (~2s).
- **Constitution note**: This *fuse-don't-gate* topology is a justified deviation from a
  strict literal reading of Principle IV ("diarizer returns audio segments to the
  transcription stage"). See `plan.md` → Complexity Tracking. The diarizer is still
  first-class and runs first on the audio; we fuse timelines instead of chopping the ASR
  input, purely to honor the explicit low-latency requirement.

## Decision 5 — Hardware-aware backend selection (M-series GPU + CPU first)

- **Decision**: A `backends/device.py` resolver picks, at load time:
  1. **Apple Silicon (default)** — ONNX Runtime **CoreML EP with `MLComputeUnits = .cpuAndGPU`**
     (Metal GPU + CPU) for the ASR; CoreML (Sortformer) for diarization. ANE (`.all`) is an
     opt-in profile, not the default.
  2. **Fallback** — ONNX Runtime **CPU EP** (and PyTorch-MPS for the reference path) when
     CoreML is unavailable or on non-Apple hardware.
  The resolved backend + compute units are recorded in the readiness report (`compute_backend`).
- **Rationale**: Constitution V requires hardware-aware, pluggable backends with CPU fallback
  only when acceleration is unavailable. Targeting the **Metal GPU + CPU** gives predictable
  FP16 throughput and full op coverage for FastConformer, avoiding the ANE op-coverage/
  palettization quirks while still using the accelerator. Keeps inference code free of device
  branching (the resolver injects the chosen runtime + compute units).
- **Alternatives considered**: Defaulting to ANE (`.all`) — deferred to an opt-in profile due
  to partial FastConformer op coverage; hard-coding CoreML — rejected (breaks FR-013 readiness
  and offline CI on non-Apple-Silicon machines).

## Decision 8 — ASR precision: FP16 default, quantization behind a measured gate

- **Decision**: Default ASR precision is **FP16**. Precision is a config knob
  (`precision: fp16 | int8 | int4`). **INT8 weight-only** is the documented fallback if memory/
  latency pressure ever appears; **INT4** is experimental only. No quantized variant is adopted
  as the default until it passes a **quantization gate**: on the multilingual fixture set
  (`tests/fixtures/audio/`), the variant must show **≤1% absolute WER regression vs FP16
  per language band** AND a **measured turn-to-text latency improvement** on the target machine.
- **Rationale**:
  - A 0.6B model at FP16 (~1.2 GB) fits the 16 GB unified-memory envelope with room for the
    diarizer, so memory is not a binding reason to quantize.
  - Weight-only INT4 on a *small* model risks uneven WER loss across the 40 locales
    (low-resource/accented speech first), and ORT INT4 matmul has weak CoreML-EP coverage
    (CPU fallback can negate the speed benefit). Quantization must therefore *earn* adoption
    with evidence, satisfying the constitution's real-time + memory gates rather than assuming
    "smaller = faster."
- **Alternatives considered**: INT4-by-default — rejected (see Decision 1); fixed FP32 —
  rejected (no quality benefit over FP16 on this model, higher memory/latency).

## Decision 6 — Model download & cache lifecycle

- **Decision**: A `models/registry.py` declares each `ModelAsset` (HF repo id, revision,
  expected files, local cache dir under the HF cache / a project models dir). A single
  `prepare()` operation downloads via `huggingface_hub` with resumable downloads and
  integrity checks; subsequent loads read from cache with no network. Progress is reported
  via callback.
- **Rationale**: Satisfies FR-011/FR-012, SC-005, and the constitution's "downloads only
  during setup, cached thereafter, CI offline after setup." Resumable + checksum addresses
  the "interrupted download" edge case.
- **Alternatives considered**: Bundling weights in the repo — rejected (size, licensing,
  update friction).

## Decision 7 — Language/version, dependencies, testing

- **Decision**: **Python 3.11**. Core deps: `onnxruntime` (CoreML EP), `coremltools`,
  `huggingface_hub`, `sounddevice` (PortAudio), `numpy`, `soxr`/`samplerate` (resampling),
  `soundfile` (fixtures). Native: **Swift** Process-Tap helper (Swift Package, built via
  `swift build`). Optional/reference: `nemo_toolkit[asr]`, `torch` (MPS) for validation.
  Testing: **pytest** with pre-recorded multi-speaker/multi-language fixtures under
  `tests/fixtures/audio/`; no network in tests; native helper covered by an integration
  test that feeds a recorded tap dump.
- **Rationale**: Matches the in-process Python library API decision, the constitution's
  dependency list (extended for ONNX/CoreML), and the offline-CI gate.
- **Alternatives considered**: Pure-PyTorch stack (heavier, slower cold start); Rust core
  (fastest but breaks the Python in-process contract) — both deferred as optimizations.

---

## Resolved unknowns summary

| Unknown | Resolution |
|---------|------------|
| ASR runtime on Apple Silicon | ONNX Runtime + CoreML EP (`.cpuAndGPU`: Metal GPU + CPU), **FP16** Nemotron export, 560ms chunk |
| ASR precision | **FP16 default**; INT8 fallback; INT4 experimental — all behind a WER+latency gate (Decision 8) |
| Diarization runtime | Streaming Sortformer CoreML, low-latency profile (~1.04s) |
| System-audio capture | Core Audio Process Taps via Swift helper (ScreenCaptureKit fallback) |
| Mic capture | PortAudio via `sounddevice`, resampled to 16 kHz mono |
| Diarize↔ASR coupling | Parallel + timestamp fusion (not gate-by-segment) — justified deviation |
| Backend selection | Hardware-aware resolver: CoreML EP GPU+CPU → CPU EP / MPS fallback |
| Model lifecycle | `huggingface_hub` resumable download + local cache, progress callback |
| Language/Version | Python 3.11 + Swift native helper |
| Testing | pytest + offline audio fixtures (incl. quantization gate) |

All NEEDS CLARIFICATION items are resolved. Ready for Phase 1.
