from dataclasses import dataclass
from typing import Literal

from henry_common.events import TelemetryEvent

from .capture import DetectionResult, SpeechChunk

type InteractionStage = Literal[
    "turn_ready",
    "transcription_completed",
    "reply_started",
    "first_reply_phrase",
    "first_audio_synthesized",
    "playback_started",
    "barge_in_detected",
    "playback_interrupted",
]


@dataclass(frozen=True, slots=True)
class InteractionTimingObserved(TelemetryEvent):
    """Elapsed interaction time measured from a completed user input."""

    stage: InteractionStage
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class TranscriptionProgressObserved(TelemetryEvent):
    content: str
    likely_complete: bool


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
