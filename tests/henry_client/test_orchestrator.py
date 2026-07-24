import asyncio

import pytest

from henry_client.events import AudioPlayed, PipelineStageChanged
from henry_client.orchestrator import ListeningMode, Orchestrator
from henry_client.pipeline import PipelineStage, PipelineStageStatus
from henry_client.reply import ReplySignal
from tests.support import (
    FakeAudioService,
    FakeReplyService,
    FakeSpeechService,
    RecordingEventSink,
    chunk,
    frame,
)


async def wait_for_event(
    sink: RecordingEventSink,
    stage: PipelineStage,
    status: PipelineStageStatus,
    count: int = 1,
) -> None:
    while (
        sum(
            isinstance(event, PipelineStageChanged)
            and event.stage is stage
            and event.status is status
            for event in sink.events
        )
        < count
    ):
        sink.changed.clear()
        await asyncio.wait_for(sink.changed.wait(), timeout=1)


class SequencedSpeechService(FakeSpeechService):
    def __init__(self, *segments) -> None:
        super().__init__()
        self._segments = iter(segments)
        self.segmented: asyncio.Queue[None] = asyncio.Queue()

    def segment(self, value, speech_detected):
        result = next(self._segments)
        self.segmented.put_nowait(None)
        return True, result


def test_orchestrator_validates_timing_and_empty_segment_limit() -> None:
    dependencies = {
        "audio": FakeAudioService(),
        "speech": FakeSpeechService(),
        "reply": FakeReplyService(),
        "events": RecordingEventSink(),
    }

    with pytest.raises(ValueError, match="delay"):
        Orchestrator(**dependencies, activation_end_delay=-0.1)
    with pytest.raises(ValueError, match="empty segments"):
        Orchestrator(**dependencies, max_empty_segments=0)


def test_orchestrator_ignores_repeated_listening_mode() -> None:
    orchestrator = Orchestrator(
        audio=FakeAudioService(),
        speech=FakeSpeechService(),
        reply=FakeReplyService(),
        events=RecordingEventSink(),
    )

    assert orchestrator._set_listening_mode(ListeningMode.PAUSED)
    assert not orchestrator._set_listening_mode(ListeningMode.PAUSED)


def test_orchestrator_plays_activation_reply_then_records() -> None:
    async def scenario() -> None:
        audio = FakeAudioService()
        speech = FakeSpeechService()
        events = RecordingEventSink()
        shutdown = asyncio.Event()
        orchestrator = Orchestrator(
            audio=audio,
            speech=speech,
            reply=FakeReplyService(activation_text="Ready."),
            events=events,
            activation_end_delay=0,
        )

        task = asyncio.create_task(orchestrator.run(shutdown))
        await wait_for_event(
            events,
            PipelineStage.LISTENING,
            PipelineStageStatus.STARTED,
        )
        await audio.chunks.put(
            chunk(
                vad_score=0.9,
                wakeword_score=0.9,
            )
        )
        await wait_for_event(
            events,
            PipelineStage.RECORDING,
            PipelineStageStatus.STARTED,
        )

        shutdown.set()
        await asyncio.wait_for(task, timeout=1)

        assert speech.synthesized[0] == "Ready."
        assert len(audio.written) == 1
        assert any(isinstance(event, AudioPlayed) for event in events.events)
        assert orchestrator._listening_mode is ListeningMode.UTTERANCE
        assert not audio.wakeword_enabled

    asyncio.run(scenario())


def test_orchestrator_runs_utterance_through_full_reply_pipeline() -> None:
    async def scenario() -> None:
        audio = FakeAudioService()
        speech = FakeSpeechService(transcript="question")
        reply = FakeReplyService("First answer.", "Second answer.")
        events = RecordingEventSink()
        shutdown = asyncio.Event()
        orchestrator = Orchestrator(
            audio=audio,
            speech=speech,
            reply=reply,
            events=events,
            activation_end_delay=0,
        )

        task = asyncio.create_task(orchestrator.run(shutdown))
        await wait_for_event(
            events,
            PipelineStage.LISTENING,
            PipelineStageStatus.STARTED,
        )
        await audio.chunks.put(chunk(vad_score=0.9, wakeword_score=0.9))
        await wait_for_event(
            events,
            PipelineStage.RECORDING,
            PipelineStageStatus.STARTED,
        )
        await audio.chunks.put(chunk(vad_score=0.9))
        await wait_for_event(
            events,
            PipelineStage.PROCESSING,
            PipelineStageStatus.COMPLETED,
        )
        await wait_for_event(
            events,
            PipelineStage.PLAYBACK,
            PipelineStageStatus.COMPLETED,
        )

        shutdown.set()
        await asyncio.wait_for(task, timeout=1)

        assert reply.received == [ReplySignal.ACTIVATION, "question"]
        assert speech.synthesized == ["First answer.", "Second answer."]
        assert len(audio.written) == 2

    asyncio.run(scenario())


def test_orchestrator_waits_after_activation_playback_before_recording() -> None:
    async def scenario() -> None:
        audio = FakeAudioService()
        speech = FakeSpeechService()
        events = RecordingEventSink()
        shutdown = asyncio.Event()
        orchestrator = Orchestrator(
            audio=audio,
            speech=speech,
            reply=FakeReplyService(activation_text="Ready."),
            events=events,
            activation_end_delay=60,
        )

        task = asyncio.create_task(orchestrator.run(shutdown))
        await wait_for_event(
            events,
            PipelineStage.LISTENING,
            PipelineStageStatus.STARTED,
        )
        await audio.chunks.put(chunk(vad_score=0.9, wakeword_score=0.9))
        await wait_for_event(
            events,
            PipelineStage.PLAYBACK,
            PipelineStageStatus.COMPLETED,
        )

        assert len(audio.written) == 1
        assert orchestrator._listening_mode is ListeningMode.PAUSED

        shutdown.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(scenario())


def test_orchestrator_returns_to_wakeword_after_empty_segment_limit() -> None:
    async def scenario() -> None:
        audio = FakeAudioService()
        speech = SequencedSpeechService(None, None)
        events = RecordingEventSink()
        shutdown = asyncio.Event()
        orchestrator = Orchestrator(
            audio=audio,
            speech=speech,
            reply=FakeReplyService(),
            events=events,
            activation_end_delay=0,
            max_empty_segments=2,
        )

        task = asyncio.create_task(orchestrator.run(shutdown))
        await wait_for_event(
            events,
            PipelineStage.LISTENING,
            PipelineStageStatus.STARTED,
        )
        await audio.chunks.put(chunk(vad_score=0.9, wakeword_score=0.9))
        await wait_for_event(
            events,
            PipelineStage.RECORDING,
            PipelineStageStatus.STARTED,
        )

        await audio.chunks.put(chunk())
        await asyncio.wait_for(speech.segmented.get(), timeout=1)
        assert orchestrator._listening_mode is ListeningMode.UTTERANCE
        assert not audio.wakeword_enabled

        await audio.chunks.put(chunk())
        await wait_for_event(
            events,
            PipelineStage.LISTENING,
            PipelineStageStatus.STARTED,
            count=2,
        )

        shutdown.set()
        await asyncio.wait_for(task, timeout=1)

        assert orchestrator._listening_mode is ListeningMode.WAKEWORD
        assert audio.wakeword_enabled
        assert audio.wakeword_resets == 2

    asyncio.run(scenario())


def test_orchestrator_resets_empty_segment_count_after_utterance() -> None:
    async def scenario() -> None:
        audio = FakeAudioService()
        speech = SequencedSpeechService(None, frame(), None, None)
        events = RecordingEventSink()
        shutdown = asyncio.Event()
        orchestrator = Orchestrator(
            audio=audio,
            speech=speech,
            reply=FakeReplyService(),
            events=events,
            activation_end_delay=0,
            max_empty_segments=2,
        )

        task = asyncio.create_task(orchestrator.run(shutdown))
        await wait_for_event(
            events,
            PipelineStage.LISTENING,
            PipelineStageStatus.STARTED,
        )
        await audio.chunks.put(chunk(vad_score=0.9, wakeword_score=0.9))
        await wait_for_event(
            events,
            PipelineStage.RECORDING,
            PipelineStageStatus.STARTED,
        )

        await audio.chunks.put(chunk())
        await asyncio.wait_for(speech.segmented.get(), timeout=1)
        await audio.chunks.put(chunk(vad_score=0.9))
        await wait_for_event(
            events,
            PipelineStage.RECORDING,
            PipelineStageStatus.STARTED,
            count=2,
        )

        await audio.chunks.put(chunk())
        await asyncio.wait_for(speech.segmented.get(), timeout=1)
        assert orchestrator._listening_mode is ListeningMode.UTTERANCE

        await audio.chunks.put(chunk())
        await wait_for_event(
            events,
            PipelineStage.LISTENING,
            PipelineStageStatus.STARTED,
            count=2,
        )

        shutdown.set()
        await asyncio.wait_for(task, timeout=1)

        assert orchestrator._listening_mode is ListeningMode.WAKEWORD

    asyncio.run(scenario())
