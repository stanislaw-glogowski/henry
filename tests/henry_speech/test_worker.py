import asyncio
import sys
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from types import SimpleNamespace
from typing import Self

import numpy as np
import pytest

import henry_speech
import henry_speech.audio as audio_package
import henry_speech.capture as capture_package
import henry_speech.synthesis as synthesis_package
import henry_speech.transcription as transcription_package
from henry_common.events import EventBus, ShutdownEvent
from henry_conversation import (
    ConversationActivated,
    GenerateReply,
    ReplyCompleted,
    ReplyLine,
    UserTurn,
)
from henry_speech.audio import AudioFormat
from henry_speech.capture import DetectionResult, SpeechChunk
from henry_speech.config import SpeechProfile, SpeechSettings
from henry_speech.segmentation import SpeechSegment
from henry_speech.synthesis import TTSProfile
from henry_speech.transcription import TranscriptionChunk, TranscriptionText
from henry_speech.worker import Worker, WorkerOptions

FORMAT = AudioFormat(sample_rate=16_000, channels=1)


def frame(value: float = 0.1):
    return FORMAT.build_frame(np.asarray([value], dtype=np.float32))


def speech_chunk(*, wakeword: bool = False) -> SpeechChunk:
    return SpeechChunk(
        audio=frame(),
        vad=DetectionResult(score=0.8, detected=True),
        wakeword=DetectionResult(score=0.9, detected=wakeword),
    )


class FakeAsyncResource(AbstractAsyncContextManager):
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> Self:
        self.entered = True
        return self

    async def __aexit__(self, *args) -> None:
        self.exited = True


class FakeCapture(FakeAsyncResource):
    def __init__(self) -> None:
        super().__init__()
        self.queue: asyncio.Queue[SpeechChunk | None] = asyncio.Queue()
        self.wakeword_enabled = False

    def enable_wakeword(self) -> None:
        self.wakeword_enabled = True

    def disable_wakeword(self) -> None:
        self.wakeword_enabled = False

    async def capture(self):
        while (item := await self.queue.get()) is not None:
            try:
                yield item
            finally:
                self.queue.task_done()


class FakeSegmentation:
    def __init__(self) -> None:
        self.reset_count = 0

    def feed(self, chunk: SpeechChunk):
        return True, SpeechSegment(chunk.audio)

    def reset(self) -> None:
        self.reset_count += 1


class FakeTranscription(FakeAsyncResource):
    async def transcribe(self, audio):
        yield TranscriptionChunk("Question")
        yield TranscriptionText("Question")


class FakeSynthesis(FakeAsyncResource):
    async def synthesize(self, text: str):
        yield frame(0.2)


class FakePlayback(FakeAsyncResource):
    def __init__(self) -> None:
        super().__init__()
        self.frames = []
        self.played = asyncio.Event()

    async def play(self, audio) -> None:
        self.frames.append(audio)
        self.played.set()


def test_worker_runs_activation_followup_synthesis_and_shutdown() -> None:
    async def scenario() -> None:
        bus = EventBus()
        capture = FakeCapture()
        segmentation = FakeSegmentation()
        transcription = FakeTranscription()
        synthesis = FakeSynthesis()
        playback = FakePlayback()
        worker = Worker(
            bus,
            capture,
            segmentation,
            transcription,
            synthesis,
            playback,
        )

        with bus.subscribe(GenerateReply) as requests:
            task = asyncio.create_task(worker.run())
            await asyncio.wait_for(_wait_until(lambda: capture.entered), 1)
            assert capture.wakeword_enabled

            capture.queue.put_nowait(speech_chunk(wakeword=True))
            activation = await asyncio.wait_for(requests.__anext__(), 1)
            requests.task_done()
            assert activation == GenerateReply(ConversationActivated())
            assert not capture.wakeword_enabled

            bus.publish(ReplyLine("Welcome"), ReplyCompleted())
            await asyncio.wait_for(playback.played.wait(), 1)
            playback.played.clear()
            assert len(playback.frames) == 1

            capture.queue.put_nowait(speech_chunk())
            turn = await asyncio.wait_for(requests.__anext__(), 1)
            requests.task_done()
            assert turn == GenerateReply(UserTurn("Question"))

            bus.publish(ReplyLine("Answer"), ReplyCompleted())
            await asyncio.wait_for(playback.played.wait(), 1)

            bus.publish(ShutdownEvent())
            await asyncio.wait_for(task, 1)

        assert all(
            resource.entered and resource.exited
            for resource in (capture, transcription, synthesis, playback)
        )
        assert segmentation.reset_count >= 1

    asyncio.run(scenario())


def test_worker_can_start_with_wakeword_disabled() -> None:
    async def scenario() -> None:
        bus = EventBus()
        capture = FakeCapture()
        worker = Worker(
            bus,
            capture,
            FakeSegmentation(),
            FakeTranscription(),
            FakeSynthesis(),
            FakePlayback(),
            WorkerOptions(wakeword_disabled=True),
        )
        task = asyncio.create_task(worker.run())
        await asyncio.wait_for(_wait_until(lambda: capture.entered), 1)
        await asyncio.wait_for(_wait_until(lambda: len(bus._subscriptions) == 1), 1)
        assert not capture.wakeword_enabled
        bus.publish(ShutdownEvent())
        await asyncio.wait_for(task, 1)

    asyncio.run(scenario())


async def _wait_until(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0)


class FakeDriver(AbstractContextManager):
    def __init__(self) -> None:
        self.input = object()
        self.output = object()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        pass

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output


def test_public_speech_runner_composes_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        driver = FakeDriver()
        vad = object()
        wakeword = object()
        tts = object()
        stt = object()
        captured: dict = {}

        monkeypatch.setattr(audio_package, "get_audio_driver", lambda _: driver)
        monkeypatch.setattr(capture_package, "get_vad_model", lambda *_: vad)
        monkeypatch.setattr(capture_package, "get_wakeword_model", lambda *_: wakeword)
        monkeypatch.setattr(synthesis_package, "get_tts_model", lambda *_: tts)
        monkeypatch.setattr(transcription_package, "get_stt_model", lambda *_: stt)

        async def fake_run(self: Worker) -> None:
            captured.update(self.__dict__)

        monkeypatch.setattr(Worker, "run", fake_run)
        profile = SpeechProfile(
            wakeword={"model": "wake.onnx"},
            tts=TTSProfile(model="voice.onnx"),
        )
        settings = SpeechSettings()
        catalog = object()
        bus = EventBus()

        await henry_speech.run_speech_worker(profile, settings, catalog, bus)
        assert captured["_event_bus"] is bus
        assert captured["_capture_service"]._audio_input is driver.input
        assert captured["_capture_service"]._vad_model is vad
        assert captured["_capture_service"]._wakeword_model is wakeword
        assert captured["_synthesis_service"]._tts_model is tts
        assert captured["_transcription_service"]._stt_model is stt

    asyncio.run(scenario())


def test_lazy_adapter_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    modules = {
        "henry_speech.audio.adapters": SimpleNamespace(
            get_audio_driver=lambda _: sentinel
        ),
        "henry_speech.capture.adapters": SimpleNamespace(
            get_vad_model=lambda *_: sentinel,
            get_wakeword_model=lambda *_: sentinel,
        ),
        "henry_speech.synthesis.adapters": SimpleNamespace(
            get_tts_model=lambda *_: sentinel
        ),
        "henry_speech.transcription.adapters": SimpleNamespace(
            get_stt_model=lambda *_: sentinel
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert audio_package.get_audio_driver(SpeechSettings().audio) is sentinel
    assert capture_package.get_vad_model(object(), SpeechSettings().vad) is sentinel
    assert (
        capture_package.get_wakeword_model(
            object(), {"model": "wake.onnx"}, SpeechSettings().wakeword
        )
        is sentinel
    )
    assert (
        synthesis_package.get_tts_model(
            TTSProfile(model="voice.onnx"), SpeechSettings().tts
        )
        is sentinel
    )
    assert (
        transcription_package.get_stt_model(
            SpeechProfile(
                wakeword={"model": "wake.onnx"},
                tts={"model": "voice.onnx"},
            ).stt,
            SpeechSettings().stt,
        )
        is sentinel
    )
