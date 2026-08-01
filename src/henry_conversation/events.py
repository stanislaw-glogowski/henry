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
class CancelReply(Event):
    """Cancel generation and describe the prefix certainly heard by the user."""

    spoken_text: str = ""


@dataclass(frozen=True, slots=True)
class ReplyGenerationStarted(Event):
    """Signal that a finite conversation graph run started generating."""

    pass


@dataclass(frozen=True, slots=True)
class ReplyGenerationCompleted(Event):
    """Signal that generation ended; audio delivery may still be active."""

    pass


@dataclass(frozen=True, slots=True)
class ReplyChunk(Event):
    text: str


@dataclass(frozen=True, slots=True)
class ReplyPhrase(Event):
    """Complete plain-text phrase ready for independent speech synthesis."""

    text: str
