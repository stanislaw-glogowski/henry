from __future__ import annotations

from pathlib import Path

import numpy as np

from henry_resources import LocalStore
from henry_speech.audio import AudioFormat
from henry_speech.capture import SpeechChunk
from henry_speech.capture.adapters import get_vad_model
from henry_speech.segmentation import UtteranceSegmenter

from .core import load_recordings, read_wav, timestamp_id, write_rows

_FRAME_SAMPLES = 512
_FORMAT = AudioFormat(sample_rate=16_000, channels=1)


def _frames(samples: np.ndarray):
    for start in range(0, len(samples), _FRAME_SAMPLES):
        values = samples[start : start + _FRAME_SAMPLES]
        if len(values) < _FRAME_SAMPLES:
            values = np.pad(values, (0, _FRAME_SAMPLES - len(values)))
        yield _FORMAT.build_frame(np.asarray(values, dtype=np.float32))


def run_endpoint(args) -> Path:
    session_path = args.session.expanduser().resolve()
    recordings = load_recordings(session_path)
    store = LocalStore()
    settings = store.load_settings().speech
    vad = get_vad_model(store, settings.vad)
    output = args.output or session_path.parents[3] / "results" / timestamp_id()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    vad.open()
    try:
        for recording in recordings:
            audio = read_wav(session_path / recording.wav_path)
            source_frames = list(_frames(audio.samples))
            silence = _FORMAT.build_frame(np.zeros(_FRAME_SAMPLES, dtype=np.float32))
            source_frames.extend([silence] * 100)
            chunks = [
                SpeechChunk(audio=frame, vad=vad.detect(frame), wakeword=None)
                for frame in source_frames
            ]
            speech_indexes = [
                index for index, chunk in enumerate(chunks) if chunk.is_speech
            ]
            segmenter = UtteranceSegmenter(settings.segmentation)
            detected_indexes: list[int] = []
            for index, chunk in enumerate(chunks):
                detected, segment = segmenter.feed(chunk)
                if detected and segment is not None:
                    detected_indexes.append(index)
            last_speech = speech_indexes[-1] if speech_indexes else None
            first_detection = detected_indexes[0] if detected_indexes else None
            rows.append(
                {
                    "sample_id": recording.sample_id,
                    "speaker_id": recording.speaker_id,
                    "condition": recording.condition,
                    "speech_detected": bool(speech_indexes),
                    "utterances_detected": len(detected_indexes),
                    "endpoint_latency_ms": (
                        round((first_detection - last_speech) * 32, 3)
                        if first_detection is not None and last_speech is not None
                        else ""
                    ),
                    "premature_endpoint": (
                        first_detection is not None
                        and last_speech is not None
                        and first_detection < last_speech
                    ),
                }
            )
    finally:
        vad.close()
    write_rows(output / "endpoint", rows)
    return output
