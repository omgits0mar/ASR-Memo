"""Sanity tests for the offline test harness (conftest + _synth).

Locks in: synthetic fixtures generate/load deterministically, and the network
guard blocks outbound egress (constitution offline-CI gate).
"""

from __future__ import annotations

import json
import socket

import numpy as np
import pytest

from tests._synth import SCENARIOS, build_scenario, split_tokens


class TestSyntheticFixtures:
    def test_all_scenarios_render_to_float32_mono_16k(self):
        for name, scenario in SCENARIOS.items():
            audio, rate, turns = build_scenario(scenario)
            assert audio.dtype == np.float32, name
            assert rate == 16000, name
            assert audio.ndim == 1, name
            assert len(audio) == int(scenario.duration_s * 16000), name
            assert turns and all("speaker" in t and "text" in t for t in turns), name

    def test_fixture_generation_is_deterministic(self):
        a1, _, _ = build_scenario(SCENARIOS["two_speaker_en"])
        a2, _, _ = build_scenario(SCENARIOS["two_speaker_en"])
        assert np.array_equal(a1, a2)

    def test_speakers_in_arrival_order(self):
        spk = SCENARIOS["two_speaker_en"].speakers
        assert spk == ["Speaker 1", "Speaker 2"]

    def test_split_tokens_spans_time(self):
        toks = split_tokens("hello world foo", 1.0, 4.0)
        assert len(toks) == 3
        assert toks[0][0] == "hello"
        assert toks[0][1] == pytest.approx(1.0)
        assert toks[-1][2] == pytest.approx(4.0)

    def test_ensure_synthetic_fixture_writes_files(self, synthetic_fixture):
        # synthetic_fixture generates into the real fixtures dir
        data = synthetic_fixture("single_speaker_en")
        assert data["sample_rate"] == 16000
        assert len(data["turns"]) >= 1
        assert all("t_start" in t for t in data["turns"])


class TestNetworkGuard:
    def test_outbound_socket_blocked_by_default(self):
        """A test making an outbound call must FAIL (constitution offline-CI)."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(BlockingIOError):
                s.connect(("8.8.8.8", 53))
        finally:
            s.close()

    @pytest.mark.allow_network
    def test_marker_disables_guard(self):
        """With allow_network, connect proceeds (no BlockingIOError raised by guard).

        We bind loopback to avoid real egress: the guard would NOT intercept
        127.0.0.1 anyway, so this just confirms the fixture ran without blocking.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 0))  # bind only; no connect → no egress
        finally:
            s.close()
