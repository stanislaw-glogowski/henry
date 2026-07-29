from abc import ABC, abstractmethod
from collections.abc import Iterator

from henry_common import AbstractResource

from ..audio import AudioFrame


class SynthesisModel(AbstractResource, ABC):
    @abstractmethod
    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        raise NotImplementedError
