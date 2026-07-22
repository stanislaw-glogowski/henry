import asyncio

import pytest

from henry_client.speech.service import SpeechService
from tests.support import FakeSTTModel, FakeTTSModel, frame


def test_speech_service_transcribes_and_streams_synthesis() -> None:
    async def scenario() -> None:
        tts = FakeTTSModel()
        service = SpeechService(stt_model=FakeSTTModel("hello"), tts_model=tts)

        async with service:
            text = await asyncio.wait_for(service.transcribe(frame()), timeout=1)
            frames = [value async for value in service.synthesize("reply")]

        assert text == "hello"
        assert len(frames) == 1
        assert frames[0].sample_rate == 22_050
        assert tts.texts == ["reply"]

    asyncio.run(scenario())


def test_speech_service_can_restart() -> None:
    async def scenario() -> None:
        service = SpeechService(stt_model=FakeSTTModel(), tts_model=FakeTTSModel())

        async with service:
            assert await service.transcribe(frame()) == "transcript"
        async with service:
            assert await service.transcribe(frame()) == "transcript"

    asyncio.run(scenario())


def test_speech_service_propagates_model_startup_failure() -> None:
    class FailingSTTModel(FakeSTTModel):
        def __enter__(self):
            raise RuntimeError("stt startup failed")

    async def scenario() -> None:
        service = SpeechService(stt_model=FailingSTTModel(), tts_model=FakeTTSModel())

        with pytest.raises(RuntimeError, match="stt startup failed"):
            async with service:
                pass

    asyncio.run(scenario())
