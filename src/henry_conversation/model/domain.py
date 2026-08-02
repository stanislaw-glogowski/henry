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
