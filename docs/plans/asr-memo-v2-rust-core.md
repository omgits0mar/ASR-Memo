# ASR-Memo v2 Master Plan: Tauri + Full Rust Core

> **Status:** Approved 2026-07-07. **Supersedes `docs/plans/cross-platform-nemo.md`**
> (NeMo/PyTorch backends inside the Python app — retired, do not implement).
> **Spec:** `docs/superpowers/specs/2026-07-07-asr-memo-v2-rust-core-design.md`
> (architecture, decisions D1–D4, components, risks — read it first).
>
> This is the phase roadmap. Each phase is implemented from its own detailed,
> bite-sized plan under `docs/superpowers/plans/`, written when the phase starts
> (so plans are grounded in the previous phase's real code, not speculation).
> Phase 1's detailed plan exists now:
> `docs/superpowers/plans/2026-07-07-v2-phase1-tauri-shell.md`.

## Why (one paragraph)

Analysis of meetily (Tauri + Rust core, whisper.cpp / Parakeet-ONNX-on-CPU, no
diarization in OSS) showed our differentiators are the models/streaming/
diarization stack, while our weaknesses are audio-pipeline robustness (the
"Teams remote voice not transcribed" bug: silent permission failure + Bluetooth
sample-rate renegotiation + naive summing mixer) and packaging/cross-platform.
v2 keeps our model stack on ONNX Runtime, rebuilds the shell and audio engine
in Rust (Tauri 2.x), and drops the NeMo/torch dependency plan entirely.

## Phases and gates

| Phase | Deliverable | Gate to pass |
|---|---|---|
| **1. Workspace + Tauri shell** | Cargo workspace; `asr-memo-core` (DTOs, traits, fakes, session skeleton); Tauri app running the existing `app/web` UI via a pywebview-shim, streaming fake segments end-to-end; 3-OS CI | App launches on macOS; `cargo test --workspace` green on 3-OS CI; UI shows live fake diarized segments |
| **2. Audio engine (Track A)** | `cpal` mic capture; macOS `cidre` global tap (tap-only aggregate, permission-silence detection); WASAPI loopback; PipeWire monitor; 50 ms alignment + RMS dynamic mix + soft-scale; persistent `rubato` resamplers; runtime rate tracking; Silero VAD gate; stream-health UI events; record-to-file mode | Teams/Zoom call on macOS recorded with **both sides audible**; pulled permission raises UI warning; loopback captures on Windows; offline pipeline tests green |
| **3. ASR port (Track B1, parallel with 2)** | `hf-hub` model manager (same cache); golden-fixture generator in Python reference; `NemotronOrtTranscriber` (3-graph cache-aware streaming RNNT on `ort`, word timestamps, language hints); EP selection CoreML/CUDA/DirectML/CPU(+int8) | Rust tokens+timestamps match Python golden fixtures within tolerances defined in the phase plan; live mic transcription on macOS + one CUDA or CPU Windows box |
| **4. Diarizer + fusion (Track B2)** | **Spike S1:** streaming-Sortformer ONNX export (FluidInference repos, else self-export from `.nemo`); `SortformerOrtDiarizer` (arrival-order labels, activation threshold); full `fusion/aligner.py` port replacing the Phase-1 midpoint aligner; session orchestrator parity | DER parity with the CoreML diarizer on validation clips; fused live transcript matches Python behavior on golden streams. *S1 fallback:* ship ASR-only on Win/Linux, diarization follows |
| **5. Services, packaging, cutover** | SQLite (`sqlx`) meetings/transcripts; full Markdown/JSON export parity; readiness UX; Tauri bundles (`.dmg`/`.msi`/`.AppImage`+`.deb`) with GPU build features + auto-detect script | `validation/` WER/DER/lang-ID ≤ macOS Python baseline; installs and runs on all 3 OSes; then archive `app/` + freeze `src/meeting_asr` as reference (meetily-style) |

Phases 2 and 3 run in parallel (maintainer decision D2). Phase 4 starts when 3's
streaming loop is stable. Phase 5 needs 2+3 (4 can trail on non-mac).

## Standing rules (all phases)

- The Python app must keep working until the Phase-5 gate passes; never break
  `make test` / `make test-fast`.
- Rust adopts the repo's offline discipline: trait-based fakes, deterministic
  tests, **no network or models in CI**; model/hardware tests behind explicit
  `#[ignore]`/features.
- Model cache stays `~/.cache/meeting_asr/models`, same pinned HF repos
  (`src/meeting_asr/models/registry.py` is the source of truth).
- JSON contracts (`SegmentDTO` etc.) are locked to `app/dto.py` shapes; the UI
  is shared, so contract drift is a bug.
- TDD, bite-sized tasks, frequent commits — per each phase's detailed plan.

## Key references

- Spec (this plan's authority): `docs/superpowers/specs/2026-07-07-asr-memo-v2-rust-core-design.md`
- Meetily patterns referenced throughout (MIT-licensed; credit in README when
  code patterns are adapted): global-tap capture & echo fix, WASAPI loopback via
  cpal, RMS dynamic mixing, persistent-resampler fix, Silero VAD gating,
  silence/health monitoring, build-time GPU features.
- Python reference implementations to port: `src/meeting_asr/asr/nemotron_onnx.py`,
  `src/meeting_asr/diarization/sortformer_coreml.py`, `src/meeting_asr/fusion/aligner.py`,
  `src/meeting_asr/session.py`, `src/meeting_asr/pipeline.py`.
