from langgraph.checkpoint.memory import InMemorySaver

from henry_common.events import EventBus

from .graph import ReplyContext, ReplyGraph, ReplyNode
from .worker import Worker

__all__ = ["run_reply_worker"]


async def run_reply_worker(
    event_bus: EventBus,
    context: ReplyContext,
) -> None:
    graph = ReplyGraph(
        node=ReplyNode(),
        checkpointer=InMemorySaver(),
    )
    await Worker(event_bus=event_bus, graph=graph, context=context).run()
