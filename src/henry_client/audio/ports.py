from abc import ABC, abstractmethod

from ..components import AbstractResource
from .domain import AudioFormat, AudioFrame


class StreamConfig(ABC):
    format: AudioFormat
    frames_per_buffer: int


class InputStream(AbstractResource, ABC):
    @abstractmethod
    def read(self) -> AudioFrame:
        raise NotImplementedError


class OutputStream(AbstractResource, ABC):
    @abstractmethod
    def write(self, frame: AudioFrame) -> None:
        raise NotImplementedError


class VADModel(AbstractResource, ABC):
    @abstractmethod
    def predict(self, frame: AudioFrame) -> float:
        raise NotImplementedError


class WakeWordModel(AbstractResource, ABC):
    @abstractmethod
    def predict(self, frame: AudioFrame) -> float:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
