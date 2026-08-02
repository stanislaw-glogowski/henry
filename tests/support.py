import threading
from collections import deque
from collections.abc import Iterator

from henry_conversation.domain import (
    LanguageModelChunk,
    LanguageModelRequest,
    LanguageModelRole,
)
from henry_conversation.model.ports import LanguageModel


class FakeLanguageModel(LanguageModel):
    def __init__(self, *responses: str) -> None:
        super().__init__()
        self.responses = deque(responses)
        self.requests: list[LanguageModelRequest] = []
        self.operations: list[tuple[str, int]] = []

    def open(self) -> None:
        self.operations.append(("open", threading.get_ident()))

    def close(self) -> None:
        self.operations.append(("close", threading.get_ident()))

    def prepare(self, role: LanguageModelRole) -> None:
        self.operations.append((f"prepare:{role}", threading.get_ident()))

    def generate(self, request: LanguageModelRequest) -> Iterator[LanguageModelChunk]:
        self.operations.append(("generate", threading.get_ident()))
        self.requests.append(request)
        for content in self.responses.popleft():
            yield LanguageModelChunk(content)
