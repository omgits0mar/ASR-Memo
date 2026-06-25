"""Diarization-accuracy gate (task T048; SC-002).

Measures speaker-attribution accuracy (DER / correct-attributed speech time)
against a labeled real multi-speaker fixture. Asserts ≥90% of speech time
attributed to the correct, stable speaker label. Needs real Sortformer
(``needs_models``); skips without it.

Uses the committed **real-speech** fixture ``real_twospeaker_en`` (LibriSpeech-
derived, CC BY 4.0): the speech-trained Sortformer reads the synthetic
carrier-tone fixtures as silence (no speaker activity), so a real clip is required
for a meaningful DER.
"""

from __future__ import annotations

import pytest

from tests._metrics import der

ATTRIBUTION_GATE = 0.90  # ≥90% of speech time correctly attributed (SC-002)


@pytest.mark.needs_models
@pytest.mark.slow
def test_diarization_attribution_accuracy(synthetic_fixture):
    pytest.importorskip("coremltools")
    from meeting_asr import AudioSourceKind, Backends, start_session
    from meeting_asr.diarization.sortformer_coreml import SortformerCoreMLDiarizer
    from meeting_asr.backends.device import default_probe, resolve_backend
    from tests._fakes import FixtureCapture, ManifestTranscriber

    manifest = synthetic_fixture("real_twospeaker_en")
    dia = SortformerCoreMLDiarizer()
    try:
        dia.load(resolve_backend(default_probe()))
    except Exception as e:
        pytest.skip(f"Sortformer not cached: {e}")

    # Reference attribution = manifest turns (speaker, [t0,t1]).
    reference = [(t["t_start"], t["t_end"], t["speaker"]) for t in manifest["turns"]]

    backends = Backends(
        capture=FixtureCapture(manifest["wav"], source=AudioSourceKind.MICROPHONE),
        diarizer=dia,
        transcriber=ManifestTranscriber(manifest["turns"]),
    )
    session = start_session(_backends=backends)
    try:
        from tests.integration.test_us1_single_source import _drain_until_stable

        _drain_until_stable(session, timeout_s=15.0)
    finally:
        final = session.stop(timeout_s=10.0)

    attributed = [(s.start, s.end, s.speaker_label) for s in final]
    error_rate = der(attributed, reference)
    assert error_rate <= 1.0 - ATTRIBUTION_GATE, (
        f"DER {error_rate:.3f} exceeds 1 - {ATTRIBUTION_GATE} (SC-002 gate)"
    )
