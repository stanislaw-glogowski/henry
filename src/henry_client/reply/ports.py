from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..components import AbstractResource
from .domain import ReplyChunk, ReplyRequest


class Responder(AbstractResource, ABC):
    @abstractmethod
    def respond(self, request: ReplyRequest) -> Iterator[ReplyChunk]:
        """Yield response chunks synchronously in the responder's worker."""
        raise NotImplementedError
