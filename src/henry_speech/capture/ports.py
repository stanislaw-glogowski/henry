from abc import ABC, abstractmethod

from henry_common import AbstractResource

from ..audio import AudioFrame


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
