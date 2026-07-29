import asyncio
from collections.abc import AsyncIterator

import pytest

from henry_speech.reply import ReplyChunk, ReplyLine, ReplyRequest
from henry_server.reply import ReplyEvent
from henry_server.session import (
    SessionClosedError,
    SessionNotFoundError,
    SessionService,
    SessionStatus,
)


class FakeReplyProvider:
    def __init__(self) -> None:
        self.requests: list[tuple[str, ReplyRequest]] = []

    async def reply(
        self,
        thread_id: str,
        request: ReplyRequest,
    ) -> AsyncIterator[ReplyEvent]:
        self.requests.append((thread_id, request))
        yield ReplyChunk(f"chunk:{request}")
        yield ReplyLine(f"line:{request}")


def test_session_processes_input_and_replays_missed_events() -> None:
    async def scenario() -> None:
        provider = FakeReplyProvider()
        service = SessionService(provider)
        session = service.create_session()

        async with session.subscribe() as subscription:
            await session.submit("hello")
            first = await asyncio.wait_for(anext(subscription), timeout=1)
            second = await asyncio.wait_for(anext(subscription), timeout=1)

        assert first.id == 1
        assert first.reply == ReplyChunk("chunk:hello")
        assert second.id == 2
        assert second.reply == ReplyLine("line:hello")
        assert provider.requests == [(session.thread_id, "hello")]

        async with session.subscribe(after_event_id=1) as replay:
            assert await asyncio.wait_for(anext(replay), timeout=1) == second

        await service.close()
        assert session.status is SessionStatus.CLOSED

    asyncio.run(scenario())


def test_closed_and_unknown_sessions_are_rejected() -> None:
    async def scenario() -> None:
        service = SessionService(FakeReplyProvider())
        session = service.create_session()

        await service.close_session(session.thread_id)

        with pytest.raises(SessionClosedError):
            await session.submit("too late")
        with pytest.raises(SessionNotFoundError):
            service.get_session(session.thread_id)
        with pytest.raises(SessionNotFoundError):
            await service.close_session(session.thread_id)

    asyncio.run(scenario())
