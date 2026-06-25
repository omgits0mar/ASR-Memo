# Contract: Public In-Process Python API

**Module**: `meeting_asr` (facade in `src/meeting_asr/__init__.py`)
**Style**: In-process Python library (the chosen API surface). Live segments delivered via
**callback** and an **iterator**; no network server.

This is the surface the future UI and tests consume. Maps to FR-010, FR-011, FR-012, FR-013,
FR-017, FR-019, FR-020.

```python
def prepare_models(
    *,
    progress: Callable[[PrepareProgress], None] | None = None,
    force: bool = False,
) -> SystemReadinessReport:
    """Ensure ASR + diarizer models are downloaded and cached. Idempotent; loads from
    cache without re-downloading when already present (FR-011, FR-012, SC-005).
    Resumable on interruption; never leaves a corrupt cache."""

def check_readiness() -> SystemReadinessReport:
    """Report model availability, mic + system-audio permissions, resolved compute
    backend, and OS capability, plus a `missing` list. Never raises for 'not ready'
    (FR-013)."""

def start_session(
    *,
    sources: Sequence[AudioSourceKind] = (AudioSourceKind.MICROPHONE, AudioSourceKind.SYSTEM),
    language_hint: str | None = None,
    on_segment: Callable[[TranscriptSegment], None] | None = None,
    on_error: Callable[[ErrorInfo], None] | None = None,
) -> "TranscriptionSession":
    """Create + start a live session. Begins capture, diarization, and transcription.
    `on_segment` fires for each finalized segment as it is produced (FR-006, FR-010).
    Raises ReadinessError if required models/permissions are missing."""


class TranscriptionSession:
    session_id: str
    status: SessionStatus

    def segments(self) -> Iterator[TranscriptSegment]:
        """Blocking iterator yielding finalized segments in order as they arrive
        (alternative to the on_segment callback)."""

    def transcript(self) -> list[TranscriptSegment]:
        """Snapshot of all finalized segments so far, in chronological order —
        available at any point during an ACTIVE session, not only after stop (FR-019)."""

    def speakers(self) -> dict[str, Speaker]:
        """Current speaker roster (stable labels)."""

    def stop(self, *, timeout_s: float = 10.0) -> list[TranscriptSegment]:
        """Halt capture/processing; flush in-flight audio; return the final transcript.
        Idempotent (FR-010)."""
```

## Behavioral contract

- **Structured results only**: every return value is a dataclass / list of dataclasses
  (no ad-hoc dicts) so a UI can bind directly (FR-017).
- **Sequential sessions**: at most one ACTIVE session at a time over the backend lifetime;
  a new `start_session` while one is ACTIVE raises `SessionBusyError`. Each session has its
  own independent transcript and speaker labeling (FR-020).
- **Permissions**: if a required source's permission is missing, `start_session` raises a
  `PermissionError` subclass with an actionable hint (mic vs system audio) (FR-015).
- **On-device**: no method performs outbound network I/O except `prepare_models` (download).
  Audio and transcript never leave the process/host (FR-014, SC-006).
- **Errors**: recoverable issues surface via `on_error` / session `status == ERROR` with
  `ErrorInfo`; the API does not crash the host process.

## Acceptance mapping

| Spec acceptance | Covered by |
|-----------------|------------|
| US1: start → labeled streamed segments → query → stop | `start_session`, `on_segment`/`segments()`, `transcript()`, `stop()` |
| US2: both sources merged; permission error | `sources=(MIC, SYSTEM)`, `PermissionError` |
| US3: per-segment language, optional hint | `TranscriptSegment.language`, `language_hint` |
| US4: prepare, readiness, lifecycle | `prepare_models`, `check_readiness`, session methods |
