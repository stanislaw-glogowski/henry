from __future__ import annotations

from pathlib import Path
from time import perf_counter

from pydantic import TypeAdapter

from henry_speech.audio import AudioBuffer
from henry_speech.synthesis.adapters import get_tts_model
from henry_speech.synthesis.config import TTSProfile, TTSSettings

from .core import benchmark_root, load_suite, timestamp_id, write_rows, write_wav


def run_tts(args) -> Path:
    suite = load_suite(args.suite)
    profile_key = "model_path" if args.adapter == "piper" else "model_id"
    profile = TTSProfile(tts={profile_key: args.model})
    settings = TypeAdapter(TTSSettings).validate_python({"adapter": args.adapter})
    model = get_tts_model(profile, settings)
    output = args.output or benchmark_root() / "results" / timestamp_id()
    output = output.expanduser().resolve()
    audio_path = output / "audio"
    rows: list[dict] = []
    opened = perf_counter()
    model.open()
    load_seconds = perf_counter() - opened
    try:
        for prompt in suite.prompts:
            buffer = AudioBuffer()
            started = perf_counter()
            first_audio_seconds: float | None = None
            for frame in model.synthesize(prompt.text):
                if first_audio_seconds is None:
                    first_audio_seconds = perf_counter() - started
                buffer.append(frame)
            inference_seconds = perf_counter() - started
            audio = buffer.build()
            if audio is None:
                raise RuntimeError(
                    f"TTS adapter produced no audio for prompt: {prompt.id}"
                )
            wav_path = audio_path / f"{prompt.id}.wav"
            write_wav(wav_path, audio)
            duration = audio.samples_count / audio.format.sample_rate
            rows.append(
                {
                    "sample_id": prompt.id,
                    "category": prompt.category,
                    "adapter": args.adapter,
                    "model": args.model,
                    "text": prompt.text,
                    "wav_path": str(wav_path.relative_to(output)),
                    "model_load_seconds": round(load_seconds, 6),
                    "first_audio_seconds": round(first_audio_seconds or 0.0, 6),
                    "inference_seconds": round(inference_seconds, 6),
                    "audio_seconds": round(duration, 6),
                    "real_time_factor": round(inference_seconds / duration, 6),
                }
            )
    finally:
        model.close()
    write_rows(output / "tts", rows)
    return output
