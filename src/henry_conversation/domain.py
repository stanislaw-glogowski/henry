from dataclasses import dataclass
from enum import StrEnum


class ConversationRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LanguageModelRole(StrEnum):
    FAST = "fast"
    DETAILED = "detailed"
    CLASSIFIER = "classifier"


class ResponseMode(StrEnum):
    FAST = "fast"
    DETAILED = "detailed"


class TurnIntent(StrEnum):
    DIRECT_RESPONSE = "direct_response"
    ACKNOWLEDGE_THEN_RESPONSE = "acknowledge_then_response"
    CLARIFY = "clarify"
    COMMAND = "command"
    CANCEL = "cancel"
    NO_RESPONSE = "no_response"


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: ConversationRole
    content: str


@dataclass(frozen=True, slots=True)
class LanguageModelRequest:
    role: LanguageModelRole
    messages: tuple[ConversationMessage, ...]


@dataclass(frozen=True, slots=True)
class LanguageModelChunk:
    content: str


@dataclass(frozen=True, slots=True)
class ResponsePlan:
    intent: TurnIntent
    mode: ResponseMode
    acknowledge: bool = False


@dataclass(frozen=True, slots=True)
class ConversationTextChunk:
    content: str
    acknowledgement: bool = False
