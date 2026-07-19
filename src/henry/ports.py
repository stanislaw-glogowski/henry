from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from .domain.audio import AudioFrame
from .domain.conversation import ConversationMessage
from .domain.speech import SpeechChunk, SpeechSegment, SpeechTranscription
from .events import AppEvent
from .telemetry import TelemetryMeasurement


class AudioInput(ABC):
    @abstractmethod
    async def read(self) -> AudioFrame:
        raise NotImplementedError


class AudioOutput(ABC):
    @abstractmethod
    async def play(self, frame: AudioFrame) -> None:
        raise NotImplementedError


class VoiceActivityDetector(ABC):
    @abstractmethod
    def analyze(self, frame: AudioFrame) -> SpeechChunk:
        raise NotImplementedError


class SpeechTranscriber(ABC):
    @abstractmethod
    async def transcribe(self, segment: SpeechSegment) -> SpeechTranscription | None:
        raise NotImplementedError


class SpeechSynthesizer(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        raise NotImplementedError


class LLModel(ABC):
    @abstractmethod
    async def generate_reply(self, messages: Sequence[ConversationMessage]) -> str:
        raise NotImplementedError


class AppEventSink(ABC):
    @abstractmethod
    def publish(self, *events: AppEvent) -> None:
        raise NotImplementedError


class TelemetrySink(ABC):
    @abstractmethod
    def publish(self, *measurements: TelemetryMeasurement) -> None:
        raise NotImplementedError
