from dataclasses import dataclass

from henry_common.events import Event


@dataclass(frozen=True, slots=True)
class ConversationActivated:
    pass


@dataclass(frozen=True, slots=True)
class UserTurn:
    text: str


type ConversationInput = ConversationActivated | UserTurn


@dataclass(frozen=True, slots=True)
class GenerateReply(Event):
    input: ConversationInput


@dataclass(frozen=True, slots=True)
class ReplyStarted(Event):
    pass


@dataclass(frozen=True, slots=True)
class ReplyCompleted(Event):
    pass


@dataclass(frozen=True, slots=True)
class ReplyChunk(Event):
    text: str


@dataclass(frozen=True, slots=True)
class ReplyLine(Event):
    text: str
