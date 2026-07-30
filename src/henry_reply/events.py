from dataclasses import dataclass

from henry_common import Event


@dataclass(frozen=True, slots=True)
class GenerateReply(Event):
    text: str | None


@dataclass(frozen=True, slots=True)
class ReplyStarted(Event):
    is_background: bool = False


class ReplyCompleted(Event): ...


@dataclass(frozen=True, slots=True)
class ReplyChunk(Event):
    text: str


@dataclass(frozen=True, slots=True)
class ReplyLine(Event):
    text: str
