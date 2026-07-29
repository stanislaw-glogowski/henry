from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppEvent: ...


@dataclass(frozen=True, slots=True)
class TelemetryEvent(AppEvent): ...


class AppEventSink(ABC):
    @abstractmethod
    def publish(self, *events: AppEvent) -> None:
        raise NotImplementedError
