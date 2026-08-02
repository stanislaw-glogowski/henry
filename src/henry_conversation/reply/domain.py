from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConversationTextChunk:
    content: str
    acknowledgement: bool = False
