from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..audio import AudioFrame
from ..components import AbstractResource


class TTSModel(AbstractResource, ABC):
    @abstractmethod
    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        raise NotImplementedError


class STTModel(AbstractResource, ABC):
    @abstractmethod
    def transcribe(self, frame: AudioFrame) -> str | None:
        raise NotImplementedError
