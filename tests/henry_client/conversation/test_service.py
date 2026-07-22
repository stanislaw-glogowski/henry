import asyncio

import pytest

from henry_client.conversation import MessageChunk, MessageLine
from henry_client.conversation.domain import MessageRole
from henry_client.conversation.service import ConversationService
from tests.support import FakeLanguageModel


def test_conversation_service_streams_chunks_lines_and_final_text() -> None:
    async def scenario() -> None:
        model = FakeLanguageModel("First", " line.\nSecond line.")
        service = ConversationService(model=model, system_prompt="system")

        async with service:
            replies = [reply async for reply in service.generate_reply("question")]

        chunks = [reply for reply in replies if type(reply) is MessageChunk]
        lines = [reply for reply in replies if isinstance(reply, MessageLine)]
        final = [reply for reply in replies if isinstance(reply, str)]

        assert [item.content for item in chunks] == ["First", " line.\nSecond line."]
        assert [item.content for item in lines] == ["First line.", "Second line."]
        assert final == ["First line.\nSecond line."]
        assert model.messages[0][0].role is MessageRole.SYSTEM

    asyncio.run(scenario())


def test_conversation_service_can_restart_and_resets_history() -> None:
    async def scenario() -> None:
        model = FakeLanguageModel("reply")
        service = ConversationService(model=model)

        async with service:
            _ = [reply async for reply in service.generate_reply("first")]
        async with service:
            _ = [reply async for reply in service.generate_reply("second")]

        assert [message.content for message in model.messages[1]] == ["second"]

    asyncio.run(scenario())


def test_conversation_service_propagates_model_startup_failure() -> None:
    class FailingLanguageModel(FakeLanguageModel):
        def __enter__(self):
            raise RuntimeError("model startup failed")

    async def scenario() -> None:
        service = ConversationService(model=FailingLanguageModel())

        with pytest.raises(RuntimeError, match="model startup failed"):
            async with service:
                pass

    asyncio.run(scenario())
