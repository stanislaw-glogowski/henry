from __future__ import annotations

from pathlib import Path
from time import perf_counter

from pydantic import TypeAdapter

from henry_speech.transcription.adapters import get_stt_model
from henry_speech.transcription.config import STTProfile, STTSettings

from .core import error_rate, load_recordings, read_wav, timestamp_id, write_rows


def run_stt(args) -> Path:
    session_path = args.session.expanduser().resolve()
    recordings = load_recordings(session_path)
    profile_values: dict[str, object] = {}
    if args.model is not None:
        profile_values["model_id"] = args.model
    if args.language is not None:
        profile_values["language"] = args.language
    profile = STTProfile(stt=profile_values)
    settings = TypeAdapter(STTSettings).validate_python({"adapter": args.adapter})
    model = get_stt_model(profile, settings)
    output = args.output or session_path.parents[3] / "results" / timestamp_id()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    opened = perf_counter()
    model.open()
    load_seconds = perf_counter() - opened
    try:
        for recording in recordings:
            frame = read_wav(session_path / recording.wav_path)
            started = perf_counter()
            chunks = list(model.transcribe(frame))
            inference_seconds = perf_counter() - started
            hypothesis = "".join(chunk.content for chunk in chunks).strip()
            rows.append(
                {
                    "sample_id": recording.sample_id,
                    "speaker_id": recording.speaker_id,
                    "condition": recording.condition,
                    "adapter": args.adapter,
                    "language": args.language or "auto",
                    "model": args.model or "adapter-default",
                    "reference": recording.reference_text,
                    "hypothesis": hypothesis,
                    "wer": round(error_rate(recording.reference_text, hypothesis), 6),
                    "cer": round(
                        error_rate(
                            recording.reference_text, hypothesis, characters=True
                        ),
                        6,
                    ),
                    "model_load_seconds": round(load_seconds, 6),
                    "inference_seconds": round(inference_seconds, 6),
                    "audio_seconds": round(recording.duration_seconds, 6),
                    "real_time_factor": round(
                        inference_seconds / recording.duration_seconds, 6
                    ),
                }
            )
    finally:
        model.close()
    write_rows(output / "stt", rows)
    return output
