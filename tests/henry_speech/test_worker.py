import asyncio
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Self

import numpy as np
import pytest

import henry_speech
import henry_speech.audio as audio_package
import henry_speech.audio.adapters as audio_adapters
import henry_speech.capture as capture_package
import henry_speech.capture.adapters as capture_adapters
import henry_speech.synthesis as synthesis_package
import henry_speech.synthesis.adapters as synthesis_adapters
import henry_speech.transcription as transcription_package
import henry_speech.transcription.adapters as transcription_adapters
from henry_common.events import EventBus, ShutdownEvent
from henry_conversation import (
    CancelReply,
    ConversationActivated,
    GenerateReply,
    ReplyGenerationCompleted,
    ReplyGenerationStarted,
    ReplyPhrase,
    UserTurn,
)
from henry_speech.audio import (
    AudioDevice,
    AudioDevices,
    AudioFormat,
    AudioPlaybackOutcome,
)
from henry_speech.capture import DetectionResult, SpeechChunk
from henry_speech.config import SpeechProfile, SpeechSettings
from henry_speech.events import (
    InteractionTimingObserved,
    ReplyPhraseDelivered,
    ReplyPhrasePlaybackStarted,
    SpeechChunkCaptured,
    TranscriptionProgressObserved,
    UserTurnCommitted,
    VADObserved,
    WakeWordObserved,
)
from henry_speech.segmentation import SpeechSegment
from henry_speech.transcription import TranscriptionChunk, TranscriptionText
from henry_speech.worker import Worker, WorkerOptions

FORMAT = AudioFormat(sample_rate=16_000, channels=1)


def frame(value: float = 0.1):
    return FORMAT.build_frame(np.asarray([value], dtype=np.float32))


def speech_chunk(*, speech: bool = True, wakeword: bool = False) -> SpeechChunk:
    return SpeechChunk(
        audio=frame(),
        vad=DetectionResult(score=0.8 if speech else 0.1, detected=speech),
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
    def __init__(self) -> None:
        super().__init__()
        self.interrupt_count = 0

    async def synthesize(self, text: str):
        yield frame(0.2)

    def interrupt(self) -> None:
        self.interrupt_count += 1


class FakePlayback(FakeAsyncResource):
    def __init__(self) -> None:
        super().__init__()
        self.frames = []
        self.played = asyncio.Event()
        self.duck_count = 0
        self.restore_count = 0
        self.interrupt_count = 0

    async def play(self, audio) -> AudioPlaybackOutcome:
        self.frames.append(audio)
        self.played.set()
        return AudioPlaybackOutcome.PLAYED

    async def interrupt(self) -> None:
        self.interrupt_count += 1

    async def duck(self) -> None:
        self.duck_count += 1

    async def restore(self) -> None:
        self.restore_count += 1


def test_capture_telemetry_is_throttled_with_immediate_detections() -> None:
    async def scenario() -> None:
        bus = EventBus()
        worker = Worker(
            bus,
            FakeCapture(),
            FakeSegmentation(),
            FakeTranscription(),
            FakeSynthesis(),
            FakePlayback(),
        )
        with bus.subscribe(
            SpeechChunkCaptured,
            VADObserved,
            WakeWordObserved,
        ) as telemetry:
            silence = speech_chunk(speech=False)
            worker._publish_capture_telemetry(silence)
            worker._publish_capture_telemetry(silence)
            assert telemetry._queue.empty()

            worker._publish_capture_telemetry(silence)
            captured = await telemetry.__anext__()
            telemetry.task_done()
            vad = await telemetry.__anext__()
            telemetry.task_done()
            wakeword = await telemetry.__anext__()
            telemetry.task_done()
            assert captured == SpeechChunkCaptured(3, False, False)
            assert vad == VADObserved(0.1, False)
            assert wakeword == WakeWordObserved(0.9, False)

            worker._publish_capture_telemetry(speech_chunk())
            captured = await telemetry.__anext__()
            telemetry.task_done()
            vad = await telemetry.__anext__()
            telemetry.task_done()
            await telemetry.__anext__()
            telemetry.task_done()
            assert captured == SpeechChunkCaptured(1, True, False)
            assert vad == VADObserved(0.8, True)

            worker._publish_capture_telemetry(speech_chunk(speech=False, wakeword=True))
            captured = await telemetry.__anext__()
            telemetry.task_done()
            await telemetry.__anext__()
            telemetry.task_done()
            wakeword = await telemetry.__anext__()
            telemetry.task_done()
            assert captured == SpeechChunkCaptured(1, False, True)
            assert wakeword == WakeWordObserved(0.9, True)

    asyncio.run(scenario())


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

        with (
            bus.subscribe(GenerateReply) as requests,
            bus.subscribe(InteractionTimingObserved) as timings,
            bus.subscribe(
                ReplyPhrasePlaybackStarted,
                ReplyPhraseDelivered,
            ) as delivery,
        ):
            task = asyncio.create_task(worker.run())
            await asyncio.wait_for(
                _wait_until(lambda: capture.entered and capture.wakeword_enabled),
                1,
            )

            # Wake-word inference may complete after VAD has fallen on this frame.
            capture.queue.put_nowait(speech_chunk(speech=False, wakeword=True))
            activation = await asyncio.wait_for(requests.__anext__(), 1)
            requests.task_done()
            assert activation == GenerateReply(ConversationActivated())
            assert not capture.wakeword_enabled
            first_timing = await asyncio.wait_for(timings.__anext__(), 1)
            timings.task_done()
            assert first_timing.stage == "turn_ready"

            bus.publish(
                ReplyGenerationStarted(1),
                ReplyPhrase(1, 1, "Welcome"),
                ReplyGenerationCompleted(1),
            )
            await asyncio.wait_for(playback.played.wait(), 1)
            playback.played.clear()
            assert len(playback.frames) == 1
            started = await asyncio.wait_for(delivery.__anext__(), 1)
            delivery.task_done()
            delivered = await asyncio.wait_for(delivery.__anext__(), 1)
            delivery.task_done()
            assert started == ReplyPhrasePlaybackStarted(1, 1)
            assert delivered == ReplyPhraseDelivered(1, 1)

            capture.queue.put_nowait(speech_chunk())
            turn = await asyncio.wait_for(requests.__anext__(), 1)
            requests.task_done()
            assert turn == GenerateReply(UserTurn("Question"))

            bus.publish(
                ReplyGenerationStarted(2),
                ReplyPhrase(2, 1, "Answer"),
                ReplyGenerationCompleted(2),
            )
            await asyncio.wait_for(playback.played.wait(), 1)
            started = await asyncio.wait_for(delivery.__anext__(), 1)
            delivery.task_done()
            delivered = await asyncio.wait_for(delivery.__anext__(), 1)
            delivery.task_done()
            assert started == ReplyPhrasePlaybackStarted(2, 1)
            assert delivered == ReplyPhraseDelivered(2, 1)

            bus.publish(ShutdownEvent())
            await asyncio.wait_for(task, 1)

        assert all(
            resource.entered and resource.exited
            for resource in (capture, transcription, synthesis, playback)
        )
        assert segmentation.reset_count == 0

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


def test_worker_interrupts_active_reply_after_sustained_speech() -> None:
    class NoSegments(FakeSegmentation):
        def feed(self, chunk: SpeechChunk):
            return False, None

    async def scenario() -> None:
        bus = EventBus()
        capture = FakeCapture()
        synthesis = FakeSynthesis()
        playback = FakePlayback()
        worker = Worker(
            bus,
            capture,
            NoSegments(),
            FakeTranscription(),
            synthesis,
            playback,
            WorkerOptions(barge_in_speech_frames=3),
        )

        with bus.subscribe(CancelReply) as cancellations:
            task = asyncio.create_task(worker.run())
            await asyncio.wait_for(_wait_until(lambda: capture.entered), 1)
            await asyncio.wait_for(_wait_until(lambda: len(bus._subscriptions) == 2), 1)
            await asyncio.wait_for(_wait_until(lambda: capture.wakeword_enabled), 1)
            capture.queue.put_nowait(speech_chunk(wakeword=True))
            await asyncio.wait_for(
                _wait_until(lambda: not capture.wakeword_enabled),
                1,
            )
            bus.publish(ReplyGenerationStarted(1), ReplyPhrase(1, 1, "Welcome"))
            await asyncio.wait_for(playback.played.wait(), 1)
            await asyncio.wait_for(
                _wait_until(lambda: worker._delivered_phrases == ["Welcome"]),
                1,
            )

            for _ in range(3):
                capture.queue.put_nowait(speech_chunk())

            assert await asyncio.wait_for(cancellations.__anext__(), 1) == CancelReply(
                "Welcome",
                1,
            )
            cancellations.task_done()
            await asyncio.wait_for(
                _wait_until(lambda: playback.interrupt_count == 1),
                1,
            )
            assert playback.duck_count == 1
            assert playback.restore_count == 1
            assert synthesis.interrupt_count == 1

            bus.publish(ShutdownEvent())
            await asyncio.wait_for(task, 1)

    asyncio.run(scenario())


def test_worker_restores_playback_after_false_barge_in() -> None:
    class NoSegments(FakeSegmentation):
        def feed(self, chunk: SpeechChunk):
            return False, None

    async def scenario() -> None:
        bus = EventBus()
        capture = FakeCapture()
        playback = FakePlayback()
        worker = Worker(
            bus,
            capture,
            NoSegments(),
            FakeTranscription(),
            FakeSynthesis(),
            playback,
            WorkerOptions(barge_in_speech_frames=3),
        )
        task = asyncio.create_task(worker.run())
        await asyncio.wait_for(_wait_until(lambda: capture.entered), 1)
        await asyncio.wait_for(_wait_until(lambda: len(bus._subscriptions) == 1), 1)
        await asyncio.wait_for(_wait_until(lambda: capture.wakeword_enabled), 1)
        capture.queue.put_nowait(speech_chunk(wakeword=True))
        await asyncio.wait_for(
            _wait_until(lambda: not capture.wakeword_enabled),
            1,
        )
        bus.publish(ReplyGenerationStarted(1), ReplyPhrase(1, 1, "Welcome"))
        await asyncio.wait_for(playback.played.wait(), 1)

        capture.queue.put_nowait(speech_chunk())
        silence = speech_chunk()
        silence = SpeechChunk(
            audio=silence.audio,
            vad=DetectionResult(score=0.1, detected=False),
            wakeword=None,
        )
        capture.queue.put_nowait(silence)
        await asyncio.wait_for(
            _wait_until(lambda: playback.restore_count == 1),
            1,
        )
        assert playback.duck_count == 1
        assert playback.interrupt_count == 0

        bus.publish(ShutdownEvent())
        await asyncio.wait_for(task, 1)

    asyncio.run(scenario())


def test_worker_options_reject_invalid_barge_in_threshold() -> None:
    with pytest.raises(ValueError, match="positive"):
        WorkerOptions(barge_in_speech_frames=0)
    with pytest.raises(ValueError, match="positive"):
        WorkerOptions(continuation_silence_frames=0)


def test_worker_combines_semantically_incomplete_transcription() -> None:
    class SpeechOnlySegmentation(FakeSegmentation):
        def feed(self, chunk: SpeechChunk):
            if not chunk.is_speech:
                return False, None
            return True, SpeechSegment(chunk.audio)

    class SequencedTranscription(FakeAsyncResource):
        def __init__(self) -> None:
            super().__init__()
            self.texts = iter(("Chcę wiedzieć, ponieważ", "to jest ważne."))
            self.completed: asyncio.Queue[None] = asyncio.Queue()

        async def transcribe(self, audio):
            text = next(self.texts)
            yield TranscriptionChunk(text)
            yield TranscriptionText(text)
            self.completed.put_nowait(None)

    async def scenario() -> None:
        bus = EventBus()
        capture = FakeCapture()
        transcription = SequencedTranscription()
        worker = Worker(
            bus,
            capture,
            SpeechOnlySegmentation(),
            transcription,
            FakeSynthesis(),
            FakePlayback(),
            WorkerOptions(wakeword_disabled=True, continuation_silence_frames=2),
        )
        with (
            bus.subscribe(GenerateReply) as requests,
            bus.subscribe(TranscriptionProgressObserved) as progress,
            bus.subscribe(UserTurnCommitted) as committed,
        ):
            task = asyncio.create_task(worker.run())
            await asyncio.wait_for(_wait_until(lambda: capture.entered), 1)
            await asyncio.wait_for(_wait_until(lambda: len(bus._subscriptions) == 4), 1)
            request = asyncio.create_task(requests.__anext__())

            capture.queue.put_nowait(speech_chunk())
            await asyncio.wait_for(transcription.completed.get(), 1)
            transcription.completed.task_done()
            assert not request.done()
            first_progress = await asyncio.wait_for(progress.__anext__(), 1)
            progress.task_done()
            assert not first_progress.likely_complete
            assert first_progress.turn_id == 1

            capture.queue.put_nowait(speech_chunk())
            assert await asyncio.wait_for(request, 1) == GenerateReply(
                UserTurn("Chcę wiedzieć, ponieważ to jest ważne.")
            )
            requests.task_done()
            committed_turn = await asyncio.wait_for(committed.__anext__(), 1)
            committed.task_done()
            assert committed_turn == UserTurnCommitted(
                1,
                "Chcę wiedzieć, ponieważ to jest ważne.",
            )

            bus.publish(ShutdownEvent())
            await asyncio.wait_for(task, 1)

    asyncio.run(scenario())


def test_worker_commits_incomplete_turn_after_silence_limit() -> None:
    class SpeechOnlySegmentation(FakeSegmentation):
        def feed(self, chunk: SpeechChunk):
            if not chunk.is_speech:
                return False, None
            return True, SpeechSegment(chunk.audio)

    class IncompleteTranscription(FakeAsyncResource):
        def __init__(self) -> None:
            super().__init__()
            self.completed = asyncio.Event()

        async def transcribe(self, audio):
            yield TranscriptionChunk("Powiem jeszcze, ale")
            yield TranscriptionText("Powiem jeszcze, ale")
            self.completed.set()

    async def scenario() -> None:
        bus = EventBus()
        capture = FakeCapture()
        transcription = IncompleteTranscription()
        worker = Worker(
            bus,
            capture,
            SpeechOnlySegmentation(),
            transcription,
            FakeSynthesis(),
            FakePlayback(),
            WorkerOptions(wakeword_disabled=True, continuation_silence_frames=2),
        )
        with bus.subscribe(GenerateReply) as requests:
            task = asyncio.create_task(worker.run())
            await asyncio.wait_for(_wait_until(lambda: capture.entered), 1)
            await asyncio.wait_for(_wait_until(lambda: len(bus._subscriptions) == 2), 1)
            capture.queue.put_nowait(speech_chunk())
            await asyncio.wait_for(transcription.completed.wait(), 1)

            silence = SpeechChunk(
                audio=frame(0),
                vad=DetectionResult(score=0.1, detected=False),
                wakeword=None,
            )
            capture.queue.put_nowait(silence)
            capture.queue.put_nowait(silence)
            assert await asyncio.wait_for(requests.__anext__(), 1) == GenerateReply(
                UserTurn("Powiem jeszcze, ale")
            )
            requests.task_done()

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
        self.devices = AudioDevices(
            input=AudioDevice("Test microphone"),
            output=AudioDevice("Test speakers"),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        pass


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
            wakeword={"label": "Wake", "model_path": "wake.onnx"},
            tts={"model_path": "voice.onnx"},
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


def test_adapter_factory_exports() -> None:
    assert audio_package.get_audio_driver is audio_adapters.get_audio_driver
    assert capture_package.get_vad_model is capture_adapters.get_vad_model
    assert capture_package.get_wakeword_model is capture_adapters.get_wakeword_model
    assert synthesis_package.get_tts_model is synthesis_adapters.get_tts_model
    assert transcription_package.get_stt_model is transcription_adapters.get_stt_model
