#!/usr/bin/env python3
"""[DEPRECATED] Batch Whisper transcription prototype — superseded by `meeting_asr`.

This file-based, non-streaming, non-diarized Whisper prototype is **superseded** by
the on-device, realtime, diarized, multilingual library in `src/meeting_asr/`
(see `specs/001-meeting-asr-backend/`). Retained for reference only (task T046).

Prefer:
    from meeting_asr import prepare_models, start_session
    prepare_models()
    session = start_session(on_segment=lambda s: print(s.speaker_label, s.text))
    session.stop()

Original docstring follows.
---------------------------------------------------------------------------

Transcribe meeting audio into timestamped minutes using Hugging Face models.

Default model: openai/whisper-large-v3-turbo (809M params, 99+ languages).
On Apple Silicon, PyTorch uses MPS when available.

Hub alternatives discovered for this project:
  - argmaxinc/whisperkit-coreml  — native CoreML, best Mac latency
  - pyannote/speaker-diarization-3.1 — speaker labels (gated; accept terms on HF first)
  - leoapolonio/AMI_Meeting_Corpus — 100h meeting benchmark corpus
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


DEFAULT_MODEL = "openai/whisper-large-v3-turbo"


def pick_device() -> tuple[int, torch.dtype]:
    if torch.cuda.is_available():
        return 0, torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return -1, torch.float32


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    return str(timedelta(seconds=total))


def transcribe(audio_path: Path, model_id: str, language: str | None) -> str:
    device, dtype = pick_device()

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=30,
        batch_size=8,
    )

    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language

    result = pipe(str(audio_path), return_timestamps=True, generate_kwargs=generate_kwargs)

    lines: list[str] = [
        f"# Meeting transcript",
        f"",
        f"**Source:** {audio_path.name}",
        f"**Model:** [{model_id}](https://huggingface.co/{model_id})",
        f"",
    ]

    chunks = result.get("chunks") or []
    if chunks:
        for chunk in chunks:
            start = chunk["timestamp"][0] or 0.0
            text = chunk["text"].strip()
            if text:
                lines.append(f"[{format_timestamp(start)}] {text}")
    else:
        lines.append(result.get("text", "").strip())

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe meeting audio to minutes.")
    parser.add_argument("audio", type=Path, help="Path to audio file (wav, mp3, m4a, …)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write markdown transcript here (default: stdout)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-l",
        "--language",
        help="ISO language code (e.g. en). Omit for auto-detect.",
    )
    args = parser.parse_args()

    if not args.audio.is_file():
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 1

    transcript = transcribe(args.audio, args.model, args.language)

    if args.output:
        args.output.write_text(transcript, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(transcript, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
