from dataclasses import dataclass

from henry_common.events import Event, StateEvent

type ReplyId = int
type PhraseId = int


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
class ConversationReady(StateEvent):
    pass


@dataclass(frozen=True, slots=True)
class CancelReply(Event):
    """Cancel generation and describe the prefix certainly heard by the user."""

    spoken_text: str = ""
    reply_id: ReplyId | None = None


@dataclass(frozen=True, slots=True)
class ReplyGenerationStarted(Event):
    """Signal that a finite conversation graph run started generating."""

    reply_id: ReplyId


@dataclass(frozen=True, slots=True)
class ReplyGenerationCompleted(Event):
    """Signal that generation ended; audio delivery may still be active."""

    reply_id: ReplyId


@dataclass(frozen=True, slots=True)
class ReplyChunk(Event):
    reply_id: ReplyId
    text: str


@dataclass(frozen=True, slots=True)
class ReplyDraftUpdated(Event):
    """Current incomplete phrase assembled from streamed model chunks."""

    reply_id: ReplyId
    text: str


@dataclass(frozen=True, slots=True)
class ReplyPhrase(Event):
    """Complete plain-text phrase ready for independent speech synthesis."""

    reply_id: ReplyId
    phrase_id: PhraseId
    text: str
