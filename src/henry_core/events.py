from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppEvent:
    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.__repr__()}"


class AppEventSink(ABC):
    @abstractmethod
    def publish(self, *events: AppEvent) -> None:
        raise NotImplementedError
