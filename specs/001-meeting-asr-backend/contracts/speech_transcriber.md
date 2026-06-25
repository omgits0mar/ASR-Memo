# Contract: SpeechTranscriber

**Module**: `meeting_asr.asr.transcriber` (protocol) → `nemotron_onnx` (backend).
**Constitution**: II (real-time/streaming), V (hardware-aware), VI (per-turn language), VII.

```python
class SpeechTranscriber(Protocol):
    def load(self, backend: ComputeBackend, *, precision: str = "fp16") -> None:
        """Load the ASR model onto the resolved backend (CoreML EP GPU+CPU → CPU/MPS).
        `precision` defaults to "fp16"; "int8"/"int4" are opt-in (quantization gate)."""

    def reset(self) -> None:
        """Clear the cache-aware streaming state for a new session."""

    def push(self, frame: AudioFrame, *, language_hint: str | None = None) -> list[AsrToken]:
        """Feed a 16 kHz mono chunk; return newly decoded tokens with timestamps,
        per-token detected language, and scores. Maintains the FastConformer encoder/conv
        cache across calls — each frame processed exactly once (FR-005, FR-016)."""

    def flush(self) -> list[AsrToken]:
        """Emit any remaining tokens at end-of-stream/stop."""

    def supported_languages(self) -> list[str]:
        """~40 language-locales."""
```

## Backend: `NemotronOnnxTranscriber`

- Model: Nemotron 3.5 ASR Streaming 0.6B **FP16 ONNX export** via ONNX Runtime
  (CoreML Execution Provider, `MLComputeUnits = .cpuAndGPU` → M-series GPU + CPU; CPU EP fallback).
- **Precision**: `precision: fp16 (default) | int8 | int4` config knob. INT8/INT4 are opt-in and
  may only become default after passing the quantization gate (≤1% absolute WER regression vs
  FP16 per language band **and** a measured latency win on the target machine). See research
  Decision 8.
- Chunk size: 560 ms (export-optimized); configurable down to 80–320 ms if profiling allows.
- Architecture: 24-layer FastConformer encoder + RNNT decoder, cache-aware streaming;
  native punctuation/capitalization.
- Language: `language_hint=None` → auto per-turn language-ID conditioning; hint biases
  recognition while still emitting per-token detected language (FR-007, FR-008, Constitution VI).

## Contract rules

- **Streaming, incremental**: tokens emitted during the session, not only at the end (FR-005).
- **Cache continuity**: audio is fed as a continuous stream; the transcriber is NOT given
  pre-cut per-speaker segments (would reset the cache; see plan Complexity Tracking).
- **Low confidence / unknown language**: flagged via token `score`/`language=None` so the
  segment carries a LOW/UNKNOWN band rather than failing (FR-018, unsupported-language edge case).
- **Silence**: produces no tokens for non-speech; no fabricated text.
- Testable offline against fixture audio with reference transcripts (WER/latency checks).
