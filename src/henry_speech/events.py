from dataclasses import dataclass

from henry_common import TelemetryEvent

from .capture import SpeechChunk


@dataclass(frozen=True, slots=True)
class SpeechChunkCaptured(TelemetryEvent):
    audio_len: int
    voice_detected: bool
    voice_score: float
    wakeword_detected: bool | None
    wakeword_score: float | None

    @staticmethod
    def from_chunk(chunk: SpeechChunk) -> SpeechChunkCaptured:
        return SpeechChunkCaptured(
            audio_len=len(chunk.audio.samples),
            voice_detected=chunk.voice_detected,
            voice_score=chunk.voice_score,
            wakeword_detected=chunk.wakeword_detected,
            wakeword_score=chunk.wakeword_score,
        )
