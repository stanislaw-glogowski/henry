from abc import ABC, abstractmethod
from collections.abc import Iterator

from henry_common.components import AbstractResource

from .domain import LanguageModelChunk, LanguageModelRequest, LanguageModelRole


class LanguageModel(AbstractResource, ABC):
    @abstractmethod
    def prepare(self, role: LanguageModelRole) -> None:
        """Load resources needed by a model role in the owning thread."""

        raise NotImplementedError

    @abstractmethod
    def generate(self, request: LanguageModelRequest) -> Iterator[LanguageModelChunk]:
        raise NotImplementedError
