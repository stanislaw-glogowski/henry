from abc import ABC, abstractmethod
from collections.abc import Iterator

from henry_common import AbstractResource

from ..audio import AudioFrame
from .domain import TranscriptionChunk


class TranscriptionModel(AbstractResource, ABC):
    @abstractmethod
    def transcribe(self, frame: AudioFrame) -> Iterator[TranscriptionChunk]:
        raise NotImplementedError
