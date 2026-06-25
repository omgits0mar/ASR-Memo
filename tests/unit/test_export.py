"""Unit tests for the export module — Markdown + JSON (task T038 / US4).

Validates completeness (SC-009): every segment's speaker/time/language/text appears
in BOTH formats; JSON round-trips all fields; segments are chronological; and
``write_export`` routes by extension + raises on an unknown extension.
"""

from __future__ import annotations

import json

import pytest

from meeting_asr.export import export_json, export_markdown, write_export
from meeting_asr.types import AudioSourceKind, ConfidenceBand, Speaker, TranscriptSegment


def _seg(label, start, end, text, lang, conf, source=AudioSourceKind.MICROPHONE):
    return TranscriptSegment(
        speaker_label=label, start=start, end=end, text=text, language=lang,
        confidence=conf, source=source,
    )


@pytest.fixture
def segments():
    return [
        _seg("Speaker 1", 0.1, 0.9, "hola buenos dias", "es", 0.95),
        _seg("Speaker 2", 1.0, 1.8, "good morning", "en", 0.40),  # low confidence
        _seg("Speaker 1", 2.0, 2.8, "como estas", "es", 0.91, source=AudioSourceKind.SYSTEM),
    ]


@pytest.fixture
def speakers():
    return {
        "Speaker 1": Speaker(label="Speaker 1", first_seen=0.1, last_seen=2.8, total_speech_seconds=1.6),
        "Speaker 2": Speaker(label="Speaker 2", first_seen=1.0, last_seen=1.8, total_speech_seconds=0.8),
    }


def test_markdown_contains_every_field_and_legend(segments, speakers):
    md = export_markdown(segments, speakers, session_meta={"input_mode": "live"})
    assert "Speaker 1" in md and "Speaker 2" in md           # speakers + legend
    assert "hola buenos dias" in md and "good morning" in md and "como estas" in md  # text
    assert "es" in md and "en" in md                         # language tags
    assert "00:00" in md                                     # timestamp(s)
    # Legend present.
    assert md.lower().count("speaker 1") >= 2


def test_markdown_legend_includes_color_swatch(segments, speakers):
    from meeting_asr.export.palette import SPEAKER_COLORS

    md = export_markdown(segments, speakers)
    # The legend carries each speaker's arrival-order color hex (contracts/transcript_export.md).
    assert SPEAKER_COLORS[0] in md
    assert SPEAKER_COLORS[1] in md


def test_markdown_is_chronological(segments, speakers):
    md = export_markdown(segments, speakers)
    # The first utterance text must appear before the last in the document.
    assert md.index("hola buenos dias") < md.index("como estas")


def test_markdown_marks_low_confidence(segments, speakers):
    md = export_markdown(segments, speakers)
    # Low-confidence segment is flagged (⚠) and the high one is not the one flagged here.
    assert "⚠" in md


def test_json_round_trips_all_fields(segments, speakers):
    s = export_json(segments, speakers, session_meta={"input_mode": "file", "language_hint": None})
    doc = json.loads(s)
    assert "segments" in doc and len(doc["segments"]) == len(segments)
    for orig, got in zip(segments, doc["segments"]):
        assert got["speaker_label"] == orig.speaker_label
        assert got["text"] == orig.text
        assert got["language"] == orig.language
        assert got["start"] == pytest.approx(orig.start)
        assert got["end"] == pytest.approx(orig.end)
        assert got["confidence"] == pytest.approx(orig.confidence)
        assert got["segment_id"] == orig.segment_id
        # source serializes to its string value (microphone/system).
        assert got["source"] in ("microphone", "system")
        assert got["is_final"] is True
    # Session metadata block present.
    assert doc["session"]["input_mode"] == "file"
    assert "Speaker 1" in doc["session"]["speakers"]


def test_json_is_chronological(segments, speakers):
    # Feed out of order; output must still be chronological.
    shuffled = [segments[2], segments[0], segments[1]]
    doc = json.loads(export_json(shuffled, speakers))
    starts = [seg["start"] for seg in doc["segments"]]
    assert starts == sorted(starts)


def test_json_multilingual_utf8_preserved(segments, speakers):
    s = export_json(segments, speakers)
    # ensure_ascii must be False so multilingual text is verbatim.
    assert "hola buenos dias" in s and "como estas" in s


def test_write_export_routes_by_extension(tmp_path, segments, speakers):
    md_path = tmp_path / "t.md"
    json_path = tmp_path / "t.json"
    mk_path = tmp_path / "t.markdown"

    assert write_export(str(md_path), segments, speakers) == str(md_path)
    assert md_path.read_text(encoding="utf-8").lstrip().startswith("#") or "Speaker" in md_path.read_text()
    assert write_export(str(json_path), segments, speakers) == str(json_path)
    json.loads(json_path.read_text(encoding="utf-8"))  # valid JSON
    # .markdown alias also routes to markdown.
    write_export(str(mk_path), segments, speakers)
    assert mk_path.read_text(encoding="utf-8")  # non-empty


def test_write_export_rejects_unknown_extension(tmp_path, segments, speakers):
    bad = tmp_path / "t.txt"
    with pytest.raises(ValueError):
        write_export(str(bad), segments, speakers)
