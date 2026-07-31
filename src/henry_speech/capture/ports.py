from abc import ABC, abstractmethod

from henry_common.components import AbstractResource

from ..audio import AudioFrame
from .domain import DetectionResult


class DetectionModel(AbstractResource, ABC):
    @abstractmethod
    def detect(self, frame: AudioFrame) -> DetectionResult:
        raise NotImplementedError


class VADModel(DetectionModel, ABC): ...


class WakeWordModel(DetectionModel, ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
