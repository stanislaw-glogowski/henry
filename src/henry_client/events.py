from abc import ABC, abstractmethod
from dataclasses import dataclass

from .pipeline import PipelineStage, PipelineStageStatus


@dataclass(frozen=True, slots=True)
class AppEvent: ...


@dataclass(frozen=True, slots=True)
class PipelineStageChanged(AppEvent):
    stage: PipelineStage
    status: PipelineStageStatus


@dataclass(frozen=True, slots=True)
class TelemetryEvent(AppEvent): ...


@dataclass(frozen=True, slots=True)
class AudioCaptured(TelemetryEvent):
    samples_count: int
    speech_score: float
    speech_detected: bool
    wakeword_score: float | None
    wakeword_detected: bool | None


@dataclass(frozen=True, slots=True)
class AudioPlayed(TelemetryEvent):
    samples_count: int


class AppEventSink(ABC):
    @abstractmethod
    def publish(self, *events: AppEvent) -> None:
        raise NotImplementedError
