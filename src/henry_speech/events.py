from dataclasses import dataclass

from henry_common import TelemetryEvent


@dataclass(frozen=True, slots=True)
class SpeechChunkCaptured(TelemetryEvent):
    audio_len: int
    voice_detected: bool
    voice_score: float
    wakeword_detected: bool | None
    wakeword_score: float | None
