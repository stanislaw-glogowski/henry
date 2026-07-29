from abc import ABC, abstractmethod

from henry_common import AbstractResource

from .domain import AudioFrame


class AudioInput(AbstractResource, ABC):
    @abstractmethod
    def read(self) -> AudioFrame:
        raise NotImplementedError


class AudioOutput(AbstractResource, ABC):
    @abstractmethod
    def write(self, frame: AudioFrame) -> None:
        raise NotImplementedError


class AudioDriver[TInput = AudioInput, TOutput = AudioOutput](AbstractResource, ABC):
    @abstractmethod
    def get_input(self) -> TInput:
        raise NotImplementedError

    @abstractmethod
    def get_output(self) -> TOutput:
        raise NotImplementedError
