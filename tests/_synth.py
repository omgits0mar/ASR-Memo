"""Deterministic synthetic meeting fixtures (task T014).

Generates speech-like audio (distinct per-speaker carriers + syllable-rate AM +
noise) with a **known manifest** of speaker turns, transcript text, and language.
This lets the *plumbing* (capture → mix → fuse → session → facade) be tested
fully offline and deterministically. Real-model WER/language-ID/DER accuracy is
gated separately under ``needs_models`` (these signals are NOT natural speech).

The manifest feeds the *fake* diarizer/transcriber backends used in tests so they
emit DiarFrames/AsrTokens matching ground truth, exercising fusion + ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

RATE = 16000  # internal pipeline rate

# Distinct carrier frequencies per speaker so even a naive separator differs them.
_SPEAKER_FREQS = (165.0, 230.0, 295.0, 360.0)


@dataclass
class Turn:
    speaker: str
    t_start: float
    t_end: float
    text: str
    language: str = "en"

    def as_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "text": self.text,
            "language": self.language,
        }


@dataclass
class Scenario:
    name: str
    duration_s: float
    turns: List[Turn] = field(default_factory=list)

    @property
    def speakers(self) -> List[str]:
        seen, out = set(), []
        for t in self.turns:
            if t.speaker not in seen:
                seen.add(t.speaker)
                out.append(t.speaker)
        return out


def _envelope(n: int, rate: int, seed: int) -> np.ndarray:
    """Syllable-rate amplitude modulation + noise → speech-like envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / rate
    syll = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))  # ~4 Hz AM
    noise = rng.standard_normal(n).astype(np.float32) * 0.15
    # smooth onset/offset to avoid clicks
    fade = min(n, int(0.02 * rate))
    env = syll + noise
    if fade > 0:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        env[:fade] *= ramp
        env[-fade:] *= ramp[::-1]
    return env.astype(np.float32)


def build_scenario(scenario: Scenario) -> Tuple[np.ndarray, int, List[dict]]:
    """Render a scenario to (audio float32 mono @ RATE, rate, manifest turns)."""
    n = int(scenario.duration_s * RATE)
    audio = np.zeros(n, dtype=np.float32)
    for turn in scenario.turns:
        try:
            spk_idx = int(turn.speaker.split()[-1]) - 1
        except (ValueError, IndexError):
            spk_idx = 0
        freq = _SPEAKER_FREQS[spk_idx % len(_SPEAKER_FREQS)]
        s = max(0, int(turn.t_start * RATE))
        e = min(n, int(turn.t_end * RATE))
        if e <= s:
            continue
        t = np.arange(e - s, dtype=np.float32) / RATE
        carrier = np.sin(2 * np.pi * freq * t)
        env = _envelope(e - s, RATE, seed=(abs(hash(turn.speaker)) % 99991) + spk_idx)
        audio[s:e] += 0.3 * carrier * env
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    return audio, RATE, [t.as_dict() for t in scenario.turns]


def split_tokens(text: str, t_start: float, t_end: float) -> List[Tuple[str, float, float]]:
    """Split reference text into word-tokens spread evenly across [t_start, t_end]."""
    words = text.split()
    if not words:
        return []
    span = max(t_end - t_start, 1e-3)
    step = span / len(words)
    return [(w, t_start + i * step, t_start + (i + 1) * step) for i, w in enumerate(words)]


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

SCENARIOS = {
    "single_speaker_en": Scenario(
        name="single_speaker_en",
        duration_s=7.0,
        turns=[
            Turn("Speaker 1", 0.2, 1.6, "hello everyone lets begin the meeting", "en"),
            Turn("Speaker 1", 2.0, 3.4, "the first item on the agenda is budget", "en"),
            Turn("Speaker 1", 3.8, 5.4, "we need to review the quarterly numbers", "en"),
        ],
    ),
    "two_speaker_en": Scenario(
        name="two_speaker_en",
        duration_s=10.0,
        turns=[
            Turn("Speaker 1", 0.2, 1.8, "good morning team", "en"),
            Turn("Speaker 2", 2.0, 3.6, "morning how are you", "en"),
            Turn("Speaker 1", 3.9, 5.6, "ready to review the roadmap", "en"),
            Turn("Speaker 2", 5.9, 7.6, "yes lets start with priorities", "en"),
            Turn("Speaker 1", 7.9, 9.4, "sounds good to me", "en"),
        ],
    ),
    "multilingual": Scenario(
        name="multilingual",
        duration_s=11.0,
        turns=[
            Turn("Speaker 1", 0.2, 2.0, "hola buenos dias a todos", "es"),
            Turn("Speaker 2", 2.3, 4.1, "good morning everyone", "en"),
            Turn("Speaker 1", 4.4, 6.2, "vamos a comenzar la reunion", "es"),
            Turn("Speaker 2", 6.5, 8.3, "lets get started with the meeting", "en"),
            Turn("Speaker 1", 8.6, 10.4, "muchas gracias", "es"),
        ],
    ),
    "overlap": Scenario(
        name="overlap",
        duration_s=9.0,
        turns=[
            Turn("Speaker 1", 0.2, 2.4, "i think we should ship the feature now", "en"),
            Turn("Speaker 2", 1.8, 3.8, "wait we need more testing first", "en"),  # overlaps speaker 1
            Turn("Speaker 1", 4.0, 5.8, "fair point lets add coverage", "en"),
            Turn("Speaker 2", 6.0, 8.4, "agreed i will write the tests today", "en"),
        ],
    ),
}
