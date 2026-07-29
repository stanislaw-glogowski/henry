import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamPart

from henry_speech.reply import ReplyChunk, ReplyLine
from henry_server.reply import ReplyService


class FakeGraph:
    def __init__(self, parts: list[StreamPart]) -> None:
        self.parts = parts
        self.input: object = None
        self.config: RunnableConfig | None = None
        self.kwargs: dict[str, Any] = {}

    async def astream(
        self,
        input: object,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamPart]:
        self.input = input
        self.config = config
        self.kwargs = kwargs
        for part in self.parts:
            yield part


def message_part(content: str, node: str = "reply") -> StreamPart:
    return {
        "type": "messages",
        "ns": (),
        "data": (
            AIMessageChunk(content=content),
            {"langgraph_node": node},
        ),
    }


def test_reply_streams_chunks_and_complete_lines() -> None:
    async def scenario() -> None:
        graph = FakeGraph(
            [
                message_part("Hello\nwor"),
                message_part("ignored", node="other"),
                message_part("ld"),
            ]
        )
        service = ReplyService(graph)

        replies = [reply async for reply in service.reply("thread-1", "How are you?")]

        assert replies == [
            ReplyChunk("Hello\nwor"),
            ReplyLine("Hello"),
            ReplyChunk("ld"),
            ReplyLine("world"),
        ]
        assert isinstance(graph.input, dict)
        assert graph.input["messages"][0].content == "How are you?"
        assert graph.config == {"configurable": {"thread_id": "thread-1"}}
        assert graph.kwargs == {
            "stream_mode": "messages",
            "version": "v2",
        }

    asyncio.run(scenario())


def test_reply_ignores_non_message_stream_parts() -> None:
    async def scenario() -> None:
        graph = FakeGraph(
            [
                {
                    "type": "updates",
                    "ns": (),
                    "data": {"reply": {}},
                }
            ]
        )

        replies = [reply async for reply in ReplyService(graph).reply("thread-1", "Hi")]

        assert replies == []

    asyncio.run(scenario())
