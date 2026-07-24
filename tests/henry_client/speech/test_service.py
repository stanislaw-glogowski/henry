import asyncio
import threading

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
        assert frames[0].format.sample_rate == 22_050
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
    async def scenario() -> None:
        service = SpeechService(
            stt_model=FakeSTTModel(
                open_error=RuntimeError("stt startup failed"),
            ),
            tts_model=FakeTTSModel(),
        )

        with pytest.raises(RuntimeError, match="stt startup failed"):
            async with service:
                pass

    asyncio.run(scenario())


def test_speech_service_keeps_models_in_their_own_threads() -> None:
    async def scenario() -> tuple[FakeSTTModel, FakeTTSModel]:
        stt = FakeSTTModel()
        tts = FakeTTSModel()
        service = SpeechService(stt_model=stt, tts_model=tts)

        async with service:
            await service.transcribe(frame())
            _ = [value async for value in service.synthesize("reply")]

        return stt, tts

    stt, tts = asyncio.run(scenario())
    event_loop_thread = threading.get_ident()

    assert stt.thread_ids
    assert len(set(stt.thread_ids)) == 1
    assert stt.thread_ids[0] != event_loop_thread
    assert tts.thread_ids
    assert len(set(tts.thread_ids)) == 1
    assert tts.thread_ids[0] != event_loop_thread


def test_speech_service_propagates_synthesis_failure() -> None:
    async def scenario() -> None:
        service = SpeechService(
            stt_model=FakeSTTModel(),
            tts_model=FakeTTSModel(
                synthesis_error=RuntimeError("synthesis failed"),
            ),
        )

        async with service:
            with pytest.raises(RuntimeError, match="synthesis failed"):
                _ = [value async for value in service.synthesize("reply")]

    asyncio.run(scenario())


def test_speech_service_propagates_transcription_failure() -> None:
    async def scenario() -> None:
        service = SpeechService(
            stt_model=FakeSTTModel(
                transcription_error=RuntimeError("transcription failed"),
            ),
            tts_model=FakeTTSModel(),
        )

        async with service:
            with pytest.raises(RuntimeError, match="transcription failed"):
                await service.transcribe(frame())

    asyncio.run(scenario())


def test_speech_service_cleans_up_stt_when_tts_startup_fails() -> None:
    async def scenario() -> None:
        stt = FakeSTTModel()
        service = SpeechService(
            stt_model=stt,
            tts_model=FakeTTSModel(
                open_error=RuntimeError("tts startup failed"),
            ),
        )

        with pytest.raises(RuntimeError, match="tts startup failed"):
            async with service:
                pass

        assert not stt.opened

    asyncio.run(scenario())
