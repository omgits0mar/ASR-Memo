from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from meeting_asr.asr.nemotron_nemo import NemotronNeMoTranscriber
from meeting_asr.diarization.sortformer_nemo import SortformerNeMoDiarizer
from meeting_asr.types import AudioFrame, AudioSourceKind


def _frame(samples: int = 1600) -> AudioFrame:
    return AudioFrame(
        pcm=np.zeros(samples, dtype=np.float32),
        t_start=0.0,
        t_end=samples / 16000,
        source=AudioSourceKind.MICROPHONE,
    )


class FakeAsrModel:
    def transcribe(self, *args, **kwargs):
        _ = args, kwargs
        return [
            SimpleNamespace(
                text="hello there <en-US>",
                timestamp={
                    "word": [
                        {"word": "hello", "start": 0.0, "end": 0.4, "score": 0.9},
                        {"word": "there", "start": 0.4, "end": 0.8, "score": 0.8},
                    ]
                },
            )
        ]


class FakeTextOnlyAsrModel:
    def transcribe(self, *args, **kwargs):
        _ = args, kwargs
        return ["hello world <en-US>"]


def test_nemo_asr_extracts_word_timestamps_and_language_tag():
    tx = NemotronNeMoTranscriber()
    tx._model = FakeAsrModel()

    toks = tx.transcribe_array(np.zeros(16000, dtype=np.float32), language="auto")

    assert [t.text for t in toks] == ["hello", "there"]
    assert [t.language for t in toks] == ["en", "en"]
    assert toks[0].t_start == 0.0
    assert toks[1].t_end == 0.8


def test_nemo_asr_falls_back_to_even_word_timestamps():
    tx = NemotronNeMoTranscriber()
    tx._model = FakeTextOnlyAsrModel()

    toks = tx.transcribe_array(np.zeros(16000, dtype=np.float32), language=None)

    assert [t.text for t in toks] == ["hello", "world"]
    assert toks[0].t_start == 0.0
    assert toks[0].t_end == 0.5
    assert toks[1].t_start == 0.5
    assert toks[1].language == "en"


def test_nemo_asr_push_does_not_reemit_existing_words():
    tx = NemotronNeMoTranscriber()
    tx._model = FakeAsrModel()

    assert len(tx.push(_frame(), language_hint="en-US")) == 2
    assert tx.push(_frame(), language_hint="en-US") == []


class FakeDiarModel:
    def diarize(self, *args, **kwargs):
        _ = args, kwargs
        probs = np.array(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.1, 0.8, 0.0, 0.0],
                [0.1, 0.2, 0.1, 0.0],
            ],
            dtype=np.float32,
        )
        return [["0.00 0.08 speaker_0"]], [probs]


class FakeSegmentDiarModel:
    def diarize(self, *args, **kwargs):
        _ = args, kwargs
        return [["0.00 0.16 speaker_1", "0.16 0.24 speaker_0"]]


def test_nemo_diarizer_decodes_probability_frames_once():
    dia = SortformerNeMoDiarizer()
    dia._model = FakeDiarModel()

    frames = dia.push(_frame())

    assert [f.speaker_label for f in frames] == ["Speaker 1", "Speaker 2"]
    assert dia.push(_frame()) == []


def test_nemo_diarizer_converts_segment_output_to_frames():
    dia = SortformerNeMoDiarizer()
    dia._model = FakeSegmentDiarModel()

    frames = dia.diarize_array(np.zeros(16000, dtype=np.float32))

    assert [f.speaker_label for f in frames[:3]] == ["Speaker 1", "Speaker 1", "Speaker 2"]
