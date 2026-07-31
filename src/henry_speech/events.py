from dataclasses import dataclass

from henry_common.events import TelemetryEvent

from .capture import DetectionResult, SpeechChunk


@dataclass(frozen=True, slots=True)
class VADObserved(TelemetryEvent, DetectionResult):
    @staticmethod
    def from_chunk(chunk: SpeechChunk) -> VADObserved:
        return VADObserved(
            score=chunk.vad.score,
            detected=chunk.vad.detected,
        )


@dataclass(frozen=True, slots=True)
class WakeWordObserved(TelemetryEvent, DetectionResult):
    @staticmethod
    def from_chunk(chunk: SpeechChunk) -> WakeWordObserved | None:
        if chunk.wakeword is None:
            return None
        return WakeWordObserved(
            score=chunk.wakeword.score,
            detected=chunk.wakeword.detected,
        )


@dataclass(frozen=True, slots=True)
class SpeechChunkCaptured(TelemetryEvent):
    samples_len: int
    is_speech: bool
    is_wakeword: bool

    @staticmethod
    def from_chunk(chunk: SpeechChunk) -> SpeechChunkCaptured:
        return SpeechChunkCaptured(
            samples_len=len(chunk.audio.samples),
            is_speech=chunk.is_speech,
            is_wakeword=chunk.is_wakeword,
        )
