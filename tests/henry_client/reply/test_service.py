import asyncio
from collections.abc import Iterator

import pytest

from henry_client.components import AbstractResource
from henry_client.reply import (
    ReplyChunk,
    ReplyLine,
    ReplyRequest,
    ReplySignal,
    ReplyText,
)
from henry_client.reply.service import ReplyService, ReplyServiceError


class FakeResponder(AbstractResource):
    def __init__(
        self,
        *parts: str,
        open_error: BaseException | None = None,
    ) -> None:
        self.parts = parts
        self.open_error = open_error
        self.requests: list[ReplyRequest] = []
        self.opened = False

    def open(self) -> None:
        if self.open_error is not None:
            raise self.open_error
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def respond(self, request: ReplyRequest) -> Iterator[ReplyChunk]:
        self.requests.append(request)
        for part in self.parts:
            yield ReplyChunk(part)


def test_reply_service_streams_chunks_lines_and_final_text() -> None:
    async def scenario() -> None:
        responder = FakeResponder("First", " line.\nSecond line.")
        service = ReplyService(responder)

        async with service:
            replies = [reply async for reply in service.reply("question")]

        chunks = [reply for reply in replies if type(reply) is ReplyChunk]
        lines = [reply for reply in replies if isinstance(reply, ReplyLine)]
        texts = [reply for reply in replies if isinstance(reply, ReplyText)]

        assert [item.content for item in chunks] == ["First", " line.\nSecond line."]
        assert [item.content for item in lines] == ["First line.", "Second line."]
        assert [item.content for item in texts] == ["First line.\nSecond line."]
        assert responder.requests == ["question"]

    asyncio.run(scenario())


def test_reply_service_forwards_signal_and_emits_empty_final_text() -> None:
    async def scenario() -> None:
        responder = FakeResponder()
        service = ReplyService(responder)

        async with service:
            replies = [reply async for reply in service.reply(ReplySignal.ACTIVATION)]

        assert replies == [ReplyText("")]
        assert responder.requests == [ReplySignal.ACTIVATION]

    asyncio.run(scenario())


def test_reply_service_propagates_responder_startup_failure() -> None:
    async def scenario() -> None:
        service = ReplyService(
            FakeResponder(open_error=RuntimeError("responder startup failed"))
        )

        with pytest.raises(RuntimeError, match="responder startup failed"):
            async with service:
                pass

    asyncio.run(scenario())


def test_reply_service_requires_open_executor_and_ignores_concurrent_reply() -> None:
    async def scenario() -> None:
        service = ReplyService(FakeResponder())

        with pytest.raises(ReplyServiceError, match="not open"):
            _ = [reply async for reply in service.reply("question")]

        async with service:
            service._replying.set()
            assert [reply async for reply in service.reply("question")] == []
            service._replying.clear()

            with pytest.raises(ReplyServiceError, match="already started"):
                await service._start()

        await service._stop()

    asyncio.run(scenario())


def test_reply_service_propagates_runtime_failure() -> None:
    class FailingResponder(FakeResponder):
        def respond(self, request: ReplyRequest) -> Iterator[ReplyChunk]:
            raise RuntimeError("response failed")
            yield

    async def scenario() -> None:
        service = ReplyService(FailingResponder())

        async with service:
            with pytest.raises(RuntimeError, match="response failed"):
                _ = [reply async for reply in service.reply("question")]

    asyncio.run(scenario())


def test_reply_service_skips_empty_lines() -> None:
    async def scenario() -> None:
        service = ReplyService(FakeResponder("\nanswer"))

        async with service:
            return [reply async for reply in service.reply("question")]

    replies = asyncio.run(scenario())

    assert [reply.content for reply in replies if isinstance(reply, ReplyLine)] == [
        "answer"
    ]
