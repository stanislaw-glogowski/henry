from dataclasses import dataclass
from enum import StrEnum


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class AssistantReply:
    text: str


class Conversation:
    def __init__(self, system_prompt: str) -> None:
        self._messages = [
            ConversationMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
            ),
        ]

    @property
    def messages(self) -> tuple[ConversationMessage, ...]:
        return tuple(self._messages)

    def add_user_message(self, content: str) -> None:
        self._messages.append(
            ConversationMessage(
                role=MessageRole.USER,
                content=content,
            ),
        )

    def add_assistant_message(self, content: str) -> None:
        self._messages.append(
            ConversationMessage(
                role=MessageRole.ASSISTANT,
                content=content,
            ),
        )
