import asyncio
import threading

import pytest

from henry_client.audio.service import AudioService, AudioServiceError
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
            chunks = service.capture()
            value = await asyncio.wait_for(anext(chunks), timeout=1)
            await service.playback(frame(sample_rate=22_050))
            await chunks.aclose()

        assert value.vad_score == 0.9
        assert value.wakeword_score == 0.8
        assert len(output_stream.frames) == 1

    asyncio.run(scenario())


def test_audio_service_can_disable_wakeword_analysis() -> None:
    async def scenario() -> None:
        input_stream = FakeInputStream()
        input_stream.feed(frame())
        service = AudioService(
            input_stream=input_stream,
            output_stream=FakeOutputStream(),
            vad_model=FakeVADModel(0.9, 0.9),
            wakeword_model=FakeWakeWordModel(0.8, 0.8),
        )

        async with service:
            chunks = service.capture()
            first = await asyncio.wait_for(anext(chunks), timeout=1)
            service.disable_wakeword()
            input_stream.feed(frame())
            while True:
                second = await asyncio.wait_for(anext(chunks), timeout=1)
                if second.wakeword_score is None:
                    break
            await chunks.aclose()

        assert first.wakeword_score == 0.8
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
            service.reset_wakeword()
            input_stream.feed(frame())
            chunks = service.capture()
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
                await asyncio.wait_for(anext(service.capture()), timeout=1)

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
                    service.playback(frame(sample_rate=22_050)),
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
            chunks = service.capture()
            await asyncio.wait_for(anext(chunks), timeout=1)
            await chunks.aclose()
        async with service:
            chunks = service.capture()
            await asyncio.wait_for(anext(chunks), timeout=1)
            await chunks.aclose()

    asyncio.run(scenario())


def test_audio_service_cleans_up_capture_when_playback_startup_fails() -> None:
    async def scenario() -> None:
        input_stream = FakeInputStream()
        vad_model = FakeVADModel()
        wakeword_model = FakeWakeWordModel()
        service = AudioService(
            input_stream=input_stream,
            output_stream=FakeOutputStream(
                open_error=RuntimeError("output startup failed"),
            ),
            vad_model=vad_model,
            wakeword_model=wakeword_model,
        )

        with pytest.raises(RuntimeError, match="output startup failed"):
            async with service:
                pass

        assert not input_stream.opened
        assert not vad_model.opened
        assert not wakeword_model.opened

    asyncio.run(scenario())


def test_audio_service_requires_open_executors() -> None:
    async def scenario() -> None:
        service = AudioService(
            input_stream=FakeInputStream(),
            output_stream=FakeOutputStream(),
            vad_model=FakeVADModel(),
            wakeword_model=FakeWakeWordModel(),
        )

        service.enable_wakeword()
        with pytest.raises(AudioServiceError, match="Capture executor is not open"):
            await anext(service.capture())
        with pytest.raises(AudioServiceError, match="Playback executor is not open"):
            await service.playback(frame(sample_rate=22_050))

        await service._stop()

    asyncio.run(scenario())


def test_audio_service_rejects_duplicate_start_and_capture() -> None:
    async def scenario() -> None:
        service = AudioService(
            input_stream=FakeInputStream(),
            output_stream=FakeOutputStream(),
            vad_model=FakeVADModel(),
            wakeword_model=FakeWakeWordModel(),
        )

        async with service:
            with pytest.raises(AudioServiceError, match="already started"):
                await service._start()

            chunks = service.capture()
            await asyncio.wait_for(anext(chunks), timeout=1)
            with pytest.raises(AudioServiceError, match="already running"):
                await anext(service.capture())
            await chunks.aclose()

    asyncio.run(scenario())


def test_audio_service_closes_input_when_vad_startup_fails() -> None:
    class FailingVADModel(FakeVADModel):
        def open(self) -> None:
            raise RuntimeError("vad startup failed")

    async def scenario() -> None:
        input_stream = FakeInputStream()
        service = AudioService(
            input_stream=input_stream,
            output_stream=FakeOutputStream(),
            vad_model=FailingVADModel(),
            wakeword_model=FakeWakeWordModel(),
        )

        with pytest.raises(RuntimeError, match="vad startup failed"):
            async with service:
                pass

        assert not input_stream.opened

    asyncio.run(scenario())
