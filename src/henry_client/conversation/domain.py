from dataclasses import dataclass
from enum import StrEnum

type MessageReply = MessageChunk | MessageLine | str | None


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class MessageChunk:
    content: str


@dataclass(frozen=True, slots=True)
class MessageLine(MessageChunk): ...


@dataclass(frozen=True, slots=True)
class Message(MessageChunk):
    role: MessageRole


class ConversationStore:
    _MAX_LEN = 5

    def __init__(self) -> None:
        self._messages: list[Message] = list()

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def add_user_message(self, content: str) -> None:
        self._messages.append(
            Message(
                role=MessageRole.USER,
                content=content,
            ),
        )
        self._apply_limit()

    def add_assistant_message(self, content: str) -> None:
        self._messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=content,
            ),
        )
        self._apply_limit()

    def reset(self) -> None:
        self._messages.clear()

    def _apply_limit(self) -> None:
        del self._messages[: -self._MAX_LEN]
