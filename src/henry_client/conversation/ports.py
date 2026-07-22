from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager

from .domain import Message, MessageChunk


class LanguageModel(AbstractContextManager, ABC):
    @abstractmethod
    def generate(self, messages: Sequence[Message]) -> Iterator[MessageChunk]:
        raise NotImplementedError
