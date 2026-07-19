from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..lifecycle import ManagedResource
from .domain import AudioFormat, AudioFrame


@dataclass(frozen=True, slots=True)
class StreamConfig(AudioFormat):
    frames_per_buffer: int


class StreamManagerError(RuntimeError): ...


class StreamManager(ManagedResource, ABC):
    @abstractmethod
    def open_input(self, config: StreamConfig) -> InputStream:
        raise NotImplementedError

    @abstractmethod
    def open_output(self, config: StreamConfig) -> OutputStream:
        raise NotImplementedError


class InputStreamError(RuntimeError): ...


class InputStream(ManagedResource, ABC):
    @abstractmethod
    def read(self) -> AudioFrame:
        raise NotImplementedError


class OutputStreamError(RuntimeError): ...


class OutputStream(ManagedResource, ABC):
    @abstractmethod
    def write(self, frame: AudioFrame) -> None:
        raise NotImplementedError
