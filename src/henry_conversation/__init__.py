from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from henry_common.events import EventBus

    from .graph import ConversationContext

from .events import (
    CancelReply,
    ConversationActivated,
    GenerateReply,
    ReplyChunk,
    ReplyGenerationCompleted,
    ReplyGenerationStarted,
    ReplyPhrase,
    UserTurn,
)

__all__ = [
    "CancelReply",
    "ConversationActivated",
    "GenerateReply",
    "ReplyChunk",
    "ReplyGenerationCompleted",
    "ReplyGenerationStarted",
    "ReplyPhrase",
    "UserTurn",
    "run_conversation_worker",
]


async def run_conversation_worker(
    event_bus: EventBus,
    context: ConversationContext,
) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from .graph import ConversationGraph, ConversationNodes
    from .worker import Worker

    graph = ConversationGraph(
        nodes=ConversationNodes(),
        checkpointer=InMemorySaver(),
    )
    await Worker(event_bus=event_bus, graph=graph, context=context).run()
