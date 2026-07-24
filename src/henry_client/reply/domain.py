from dataclasses import dataclass
from enum import StrEnum

type ReplyRequest = str | ReplySignal
type Reply = ReplyChunk | ReplyLine | ReplyText


class ReplySignal(StrEnum):
    ACTIVATION = "ACTIVATION"


@dataclass(frozen=True, slots=True)
class ReplyContent:
    content: str


@dataclass(frozen=True, slots=True)
class ReplyChunk(ReplyContent):
    """Raw fragment emitted by a responder."""


@dataclass(frozen=True, slots=True)
class ReplyLine(ReplyContent):
    """Complete non-empty line ready for speech synthesis."""


@dataclass(frozen=True, slots=True)
class ReplyText(ReplyContent):
    """Complete text accumulated from all response chunks."""
