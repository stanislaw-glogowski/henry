from typing import Literal

from langgraph.graph import MessagesState

type ConversationInputKind = Literal["activation", "user_turn"]


class ConversationState(MessagesState):
    input_kind: ConversationInputKind
    summary: str
