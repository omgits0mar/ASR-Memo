"""Unit tests for the Nemotron ASR language-prompt resolution (auto-detect + hints).

Pure logic over the prompt dictionary — no ONNX model, no cache. Exercises the
language-slot selection: an explicit hint (`ar`/`en`/locale) picks that slot; an
empty/None hint falls back to the model's auto-detect slot (multilingual), so a
user who switches between English and Arabic mid-session is recognized either way.
"""

from __future__ import annotations

from meeting_asr.asr.nemotron_onnx import NemotronOnnxTranscriber


def _transcriber_with_prompts() -> NemotronOnnxTranscriber:
    t = NemotronOnnxTranscriber()
    # Mirror the real languages.json shape (subset).
    t._prompt_dict = {"en-US": 1, "en": 1, "ar-AR": 7, "ar": 7, "fr": 9}
    t._auto_slot = 101  # autoSlot in languages.json (multilingual auto-detect)
    t._num_prompts = 128
    return t


def test_explicit_hint_selects_language_slot():
    t = _transcriber_with_prompts()
    t._set_language("ar")
    assert t._lang_slot == 7
    assert t._language == "ar"

    t._set_language("en-GB")  # locale variant resolves to the base language
    assert t._lang_slot == 1
    assert t._language == "en"


def test_empty_hint_falls_back_to_auto_detect():
    t = _transcriber_with_prompts()
    t._set_language(None)
    assert t._lang_slot == 101, "no hint → multilingual auto-detect slot"
    # On auto-detect we don't assert a specific language up front.
    assert t._language is None


def test_auto_keyword_selects_auto_slot():
    t = _transcriber_with_prompts()
    t._set_language("auto")
    assert t._lang_slot == 101
    assert t._language is None


def test_language_mask_is_one_hot_on_selected_slot():
    t = _transcriber_with_prompts()
    t._set_language("ar")
    mask = t._language_mask()
    assert mask.shape == (1, 128)
    assert mask[0, 7] == 1.0
    assert mask.sum() == 1.0
