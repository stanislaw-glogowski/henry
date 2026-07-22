from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

from .domain import AudioFrame


class InputStream(AbstractContextManager, ABC):
    @abstractmethod
    def read(self) -> AudioFrame:
        raise NotImplementedError


class OutputStream(AbstractContextManager, ABC):
    @abstractmethod
    def write(self, frame: AudioFrame) -> None:
        raise NotImplementedError


class VADModel(AbstractContextManager, ABC):
    @abstractmethod
    def predict(self, frame: AudioFrame) -> float:
        raise NotImplementedError


class WakeWordModel(AbstractContextManager, ABC):
    @abstractmethod
    def predict(self, frame: AudioFrame) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
