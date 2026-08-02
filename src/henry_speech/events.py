from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

from henry_common.events import StateEvent, TelemetryEvent
from henry_conversation.events import PhraseId, ReplyId

from .audio import AudioDevices
from .capture import DetectionResult, SpeechChunk

type TurnId = int


class VoiceSessionMode(Enum):
    WAITING_FOR_WAKE_WORD = auto()
    ACTIVE = auto()


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
    turn_id: TurnId
    content: str
    likely_complete: bool


@dataclass(frozen=True, slots=True)
class UserTurnCommitted(StateEvent):
    turn_id: TurnId
    text: str


@dataclass(frozen=True, slots=True)
class ReplyPhrasePlaybackStarted(StateEvent):
    reply_id: ReplyId
    phrase_id: PhraseId


@dataclass(frozen=True, slots=True)
class ReplyPhraseDelivered(StateEvent):
    reply_id: ReplyId
    phrase_id: PhraseId


@dataclass(frozen=True, slots=True)
class AudioDevicesSelected(StateEvent):
    driver: str
    devices: AudioDevices


@dataclass(frozen=True, slots=True)
class SpeechReady(StateEvent):
    pass


@dataclass(frozen=True, slots=True)
class VoiceSessionModeChanged(StateEvent):
    mode: VoiceSessionMode


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
