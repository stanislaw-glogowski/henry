from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager

from ..audio import AudioFrame


class TTSModel(AbstractContextManager, ABC):
    @abstractmethod
    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        raise NotImplementedError


class STTModel(AbstractContextManager, ABC):
    @abstractmethod
    def transcribe(self, frame: AudioFrame) -> str | None:
        raise NotImplementedError
