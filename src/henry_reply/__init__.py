from henry_common.events import EventBus

from .graph import ReplyGraph, ReplyNode
from .worker import Worker

__all__ = ["run_reply_worker"]


async def run_reply_worker(
    event_bus: EventBus,
) -> None:

    await Worker(event_bus=event_bus, graph=ReplyGraph(node=ReplyNode())).run()
