"""Unit tests for fusion/aligner.py (T022 + T044 scope).

The aligner fuses the DiarFrame timeline with the AsrToken stream into
TranscriptSegments. Pure logic — fully offline, no models.
"""

from __future__ import annotations

import pytest

from meeting_asr.fusion.aligner import Aligner
from meeting_asr.types import AsrToken, DiarFrame


def _tok(text, t0, t1, lang="en", score=0.95):
    return AsrToken(text=text, t_start=t0, t_end=t1, language=lang, score=score)


def _frame(t0, t1, spk, score=0.9):
    return DiarFrame(t_start=t0, t_end=t1, speaker_label=spk, score=score)


def _finals(*emissions):
    """Collect only the FINAL segments across emissions (drop provisional/live
    updates), de-duped by id keeping the last occurrence — mirrors how a consumer
    that replaces-by-id ends up with the finalized transcript."""
    by_id = {}
    for batch in emissions:
        for s in batch:
            by_id[s.segment_id] = s
    return [s for s in by_id.values() if s.is_final]


class TestDominantAttribution:
    def test_single_speaker_one_segment(self):
        a = Aligner()
        a.push_diar([_frame(0.0, 3.0, "Speaker 1")])
        segs = a.push_tokens([_tok("hello", 0.1, 0.5), _tok("world", 0.6, 1.0)])
        # Continuous run streams live (provisional), then flush finalizes the line.
        final = _finals(segs, a.flush())
        assert len(final) == 1
        s = final[0]
        assert s.speaker_label == "Speaker 1"
        assert s.text == "hello world"
        assert s.start == pytest.approx(0.1) and s.end == pytest.approx(1.0)
        assert s.language == "en"
        assert s.confidence > 0.0

    def test_speaker_change_creates_two_segments(self):
        a = Aligner()
        a.push_diar([_frame(0.0, 2.0, "Speaker 1"), _frame(2.0, 4.0, "Speaker 2")])
        # token at 2.5 attributed to Speaker 2; the run closes on speaker change
        segs = a.push_tokens([
            _tok("a", 0.1, 0.4), _tok("b", 0.5, 0.8),
            _tok("c", 2.1, 2.5), _tok("d", 2.6, 3.0),
        ])
        final = sorted(_finals(segs, a.flush()), key=lambda s: s.start)
        assert len(final) == 2
        assert final[0].speaker_label == "Speaker 1" and final[0].text == "a b"
        assert final[1].speaker_label == "Speaker 2" and final[1].text == "c d"

    def test_dominant_overlap_picks_more_overlapping_speaker(self):
        a = Aligner()
        # window [1.0,1.5] overlaps Speaker 1 (1.0-1.2, 0.2s) and Speaker 2 (1.2-2.0, 0.3s)
        a.push_diar([_frame(1.0, 1.2, "Speaker 1"), _frame(1.2, 2.0, "Speaker 2")])
        segs = a.push_tokens([_tok("x", 1.0, 1.5)])
        final = _finals(segs, a.flush())
        assert final[0].speaker_label == "Speaker 2"  # greater overlap

    def test_gap_breaks_segment_same_speaker(self):
        a = Aligner(segment_gap_s=0.5)
        a.push_diar([_frame(0.0, 5.0, "Speaker 1")])
        segs = a.push_tokens([_tok("p", 0.1, 0.4), _tok("q", 2.0, 2.4)])  # 1.6s gap
        final = _finals(segs, a.flush())
        assert len(final) == 2


class TestBuffering:
    def test_tokens_beyond_diar_frontier_stream_provisionally_then_finalize(self):
        a = Aligner()
        a.push_diar([_frame(0.0, 1.0, "Speaker 1")])  # frontier = 1.0
        # Token beyond the frontier streams as a provisional update (live), not held
        # silently — so the line appears while the diarizer catches up.
        segs = a.push_tokens([_tok("late", 1.2, 1.5)])
        assert segs and all(not s.is_final for s in segs)
        assert segs[-1].text == "late"
        # Once the frontier advances, the run finalizes with the real attribution.
        a.push_diar([_frame(1.0, 2.0, "Speaker 1")])
        segs2 = a.push_tokens([])
        final = _finals(segs, segs2, a.flush())
        assert len(final) == 1
        assert final[0].text == "late" and final[0].speaker_label == "Speaker 1"

    def test_flush_finalizes_remaining(self):
        a = Aligner()
        a.push_diar([_frame(0.0, 5.0, "Speaker 1")])
        a.push_tokens([_tok("only", 0.2, 0.6)])  # open run, not closed
        assert a.flush()[0].text == "only"

    def test_empty_text_tokens_ignored(self):
        a = Aligner()
        a.push_diar([_frame(0.0, 2.0, "Speaker 1")])
        a.push_tokens([_tok("", 0.1, 0.4), _tok("real", 0.5, 0.9), _tok("   ", 1.0, 1.3)])
        final = a.flush()
        assert len(final) == 1
        assert final[0].text.strip() == "real"


class TestLiveProvisionalSegments:
    """Live behavior: a continuous same-speaker run streams as ONE growing,
    in-place-updated provisional segment (stable id, is_final=False), and is
    re-emitted final when the run closes — so the UI shows lines live without
    waiting for Stop, and natural inter-phrase pauses don't fragment a sentence.
    """

    def test_continuous_run_emits_growing_provisional_same_id(self):
        a = Aligner()
        a.push_diar([_frame(0.0, 5.0, "Speaker 1")])
        # Two pushes within one continuous run (small 0.2s inter-phrase gap).
        s1 = a.push_tokens([_tok("Hello", 0.1, 0.5)])
        s2 = a.push_tokens([_tok("there", 0.7, 1.1)])
        assert s1 and s2, "each push should emit a provisional update"
        assert all(not s.is_final for s in s1 + s2), "open run is provisional"
        # Same stable id → UI updates the same line in place, growing the text.
        assert s1[-1].segment_id == s2[-1].segment_id
        assert s1[-1].text == "Hello"
        assert s2[-1].text == "Hello there"
        # Closing the stream finalizes the one merged segment.
        final = a.flush()
        assert len(final) == 1 and final[0].is_final
        assert final[0].text == "Hello there"
        assert final[0].segment_id == s2[-1].segment_id

    def test_natural_pause_does_not_split_sentence(self):
        # Inter-phrase pauses up to the default gap stay in ONE segment.
        a = Aligner()
        a.push_diar([_frame(0.0, 12.0, "Speaker 1")])
        toks = [
            _tok("Hello", 5.0, 5.4),
            _tok("I'm", 6.0, 6.3),          # 0.6s gap
            _tok("twenty", 6.4, 6.8),
            _tok("five", 6.9, 7.2),
            _tok("years", 7.3, 7.6),
            _tok("old", 8.0, 8.3),          # 0.4s gap
        ]
        for t in toks:
            a.push_tokens([t])
        final = a.flush()
        assert len(final) == 1, f"expected one merged sentence, got {[s.text for s in final]}"
        assert final[0].text == "Hello I'm twenty five years old"

    def test_unknown_speaker_continues_current_run(self):
        # A token whose window the diarizer can't attribute ('Speaker ?') must
        # not break a run nor relabel — it continues the established speaker.
        a = Aligner()
        a.push_diar([_frame(0.0, 1.0, "Speaker 1")])  # frontier 1.0; gap after
        toks = [_tok("and", 0.1, 0.4), _tok("I", 0.5, 0.8),
                _tok("graduated", 1.2, 1.6)]  # 1.2-1.6 beyond frontier → unknown
        for t in toks:
            a.push_tokens([t])
        final = a.flush()
        assert len(final) == 1
        assert final[0].speaker_label == "Speaker 1"
        assert final[0].text == "and I graduated"
        assert "Speaker ?" not in final[0].speaker_label

    def test_real_gap_still_splits(self):
        # A genuinely large gap (well beyond inter-phrase) still starts a new line.
        # ("first" finalizes when the gap is seen on the 2nd push, not at flush.)
        a = Aligner()
        a.push_diar([_frame(0.0, 20.0, "Speaker 1")])
        s1 = a.push_tokens([_tok("first", 0.1, 0.5)])
        s2 = a.push_tokens([_tok("second", 10.0, 10.4)])  # 9.5s gap → new segment
        final = sorted(_finals(s1, s2, a.flush()), key=lambda s: s.start)
        assert len(final) == 2
        assert final[0].text == "first" and final[1].text == "second"
        assert final[0].segment_id != final[1].segment_id


class TestOrdering:
    def test_segments_emitted_in_start_order(self):
        a = Aligner()
        a.push_diar([
            _frame(0.0, 1.5, "Speaker 1"), _frame(1.5, 3.0, "Speaker 2"),
            _frame(3.0, 4.5, "Speaker 1"),
        ])
        segs = a.push_tokens([
            _tok("a1", 0.1, 0.4), _tok("a2", 0.5, 0.8),
            _tok("b1", 1.6, 1.9), _tok("b2", 2.0, 2.3),
            _tok("c1", 3.1, 3.4),
        ])
        final = sorted(_finals(segs, a.flush()), key=lambda s: s.start)
        starts = [s.start for s in final]
        assert starts == sorted(starts)
        labels = [s.speaker_label for s in final]
        assert labels[0] == "Speaker 1" and labels[1] == "Speaker 2" and labels[2] == "Speaker 1"
