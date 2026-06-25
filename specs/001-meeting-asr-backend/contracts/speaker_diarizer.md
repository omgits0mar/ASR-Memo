# Contract: SpeakerDiarizer

**Module**: `meeting_asr.diarization.diarizer` (protocol) → `sortformer_coreml` (backend).
**Constitution**: IV (first-class diarization, stable session IDs), V (hardware-aware), VII.

```python
class SpeakerDiarizer(Protocol):
    def load(self, backend: ComputeBackend) -> None:
        """Load the diarization model onto the resolved backend (CoreML → MPS/CPU)."""

    def reset(self) -> None:
        """Clear the arrival-order speaker cache (AOSC) for a new session.
        Speaker labels restart at 'Speaker 1'."""

    def push(self, frame: AudioFrame) -> list[DiarFrame]:
        """Feed a 16 kHz mono audio chunk; return any newly-resolved frame-level
        diarization decisions (80 ms granularity) with stable speaker labels.
        Streaming/online — maintains internal state across calls (FR-003, FR-004)."""

    def max_speakers(self) -> int:
        """Concurrent-speaker capacity (4 for Sortformer 4spk)."""
```

## Backend: `SortformerCoreMLDiarizer`

- Model: `nvidia/diar_streaming_sortformer_4spk-v2.1` via CoreML build
  (`FluidInference/diar-streaming-sortformer-coreml`).
- Config (low-latency profile): chunk size 6, right context 7, FIFO 188, 80 ms frames,
  16 kHz mono → ~1.04s latency, RTF ≈ 0.093.
- AOSC assigns arrival-order labels ("Speaker 1", 2, …) that remain stable for the session.

## Contract rules

- **Stable IDs**: a given voice keeps its label for the whole session; long monologues do not
  spawn new labels; a late-arriving voice gets a fresh label (edge cases).
- **Overlap**: may report multiple active speakers for the same time window.
- **Capacity exceeded**: when concurrent speakers > `max_speakers()`, attribute best-effort and
  mark affected frames low-confidence rather than failing (FR-018).
- **Output is advisory to fusion**: `DiarFrame`s feed `fusion.aligner`; the diarizer does NOT
  receive or re-segment ASR input (see plan Complexity Tracking — parallel topology).
- Testable offline against fixture audio with known speaker turns.
