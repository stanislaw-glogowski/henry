from abc import ABC, abstractmethod
from collections.abc import Iterator

from henry_common.components import AbstractResource

from ..audio import AudioFrame


class TTSModel(AbstractResource, ABC):
    @abstractmethod
    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        raise NotImplementedError
