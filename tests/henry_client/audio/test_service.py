import asyncio
import threading

import pytest

from henry_client.audio.service import AudioService
from tests.support import (
    FakeInputStream,
    FakeOutputStream,
    FakeVADModel,
    FakeWakeWordModel,
    frame,
)


def test_audio_service_reads_analysis_and_writes_frames() -> None:
    async def scenario() -> None:
        input_stream = FakeInputStream()
        output_stream = FakeOutputStream()
        wakeword = FakeWakeWordModel(0.8)
        input_stream.feed(frame(0.25))
        service = AudioService(
            input_stream=input_stream,
            output_stream=output_stream,
            vad_model=FakeVADModel(0.9),
            wakeword_model=wakeword,
        )

        async with service:
            chunks = service.read()
            value = await asyncio.wait_for(anext(chunks), timeout=1)
            await service.write(frame(sample_rate=22_050))
            await chunks.aclose()

        assert value.speech_detected
        assert value.wakeword_detected
        assert len(output_stream.frames) == 1

    asyncio.run(scenario())


def test_audio_service_disables_wakeword_after_detection() -> None:
    async def scenario() -> None:
        input_stream = FakeInputStream()
        input_stream.feed(frame(), frame())
        service = AudioService(
            input_stream=input_stream,
            output_stream=FakeOutputStream(),
            vad_model=FakeVADModel(0.9, 0.9),
            wakeword_model=FakeWakeWordModel(0.8, 0.8),
        )

        async with service:
            chunks = service.read()
            first = await asyncio.wait_for(anext(chunks), timeout=1)
            second = await asyncio.wait_for(anext(chunks), timeout=1)
            await chunks.aclose()

        assert first.wakeword_detected is True
        assert second.wakeword_detected is None
        assert second.wakeword_score is None

    asyncio.run(scenario())


def test_audio_service_resets_wakeword_in_input_worker() -> None:
    async def scenario() -> None:
        input_stream = FakeInputStream()
        wakeword = FakeWakeWordModel(0.0)
        service = AudioService(
            input_stream=input_stream,
            output_stream=FakeOutputStream(),
            vad_model=FakeVADModel(0.0),
            wakeword_model=wakeword,
        )

        async with service:
            service.enable_wakeword()
            input_stream.feed(frame())
            chunks = service.read()
            await asyncio.wait_for(anext(chunks), timeout=1)
            await asyncio.wait_for(
                asyncio.to_thread(wakeword.reset_event.wait),
                timeout=1,
            )
            await chunks.aclose()

        assert wakeword.reset_threads
        assert wakeword.reset_threads[0] != threading.get_ident()

    asyncio.run(scenario())


def test_audio_service_propagates_input_failure() -> None:
    async def scenario() -> None:
        input_stream = FakeInputStream()
        input_stream.feed(RuntimeError("read failed"))
        service = AudioService(
            input_stream=input_stream,
            output_stream=FakeOutputStream(),
            vad_model=FakeVADModel(),
            wakeword_model=FakeWakeWordModel(),
        )

        async with service:
            with pytest.raises(RuntimeError, match="read failed"):
                await asyncio.wait_for(anext(service.read()), timeout=1)

    asyncio.run(scenario())


def test_audio_service_propagates_output_failure() -> None:
    async def scenario() -> None:
        service = AudioService(
            input_stream=FakeInputStream(),
            output_stream=FakeOutputStream(RuntimeError("write failed")),
            vad_model=FakeVADModel(),
            wakeword_model=FakeWakeWordModel(),
        )

        async with service:
            with pytest.raises(RuntimeError, match="write failed"):
                await asyncio.wait_for(
                    service.write(frame(sample_rate=22_050)),
                    timeout=1,
                )

    asyncio.run(scenario())


def test_audio_service_can_restart() -> None:
    async def scenario() -> None:
        service = AudioService(
            input_stream=FakeInputStream(),
            output_stream=FakeOutputStream(),
            vad_model=FakeVADModel(),
            wakeword_model=FakeWakeWordModel(),
        )

        async with service:
            pass
        async with service:
            pass

    asyncio.run(scenario())
