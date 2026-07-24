import asyncio

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
)


async def wait_for_event(
    sink: RecordingEventSink,
    stage: PipelineStage,
    status: PipelineStageStatus,
) -> None:
    while not any(
        isinstance(event, PipelineStageChanged)
        and event.stage is stage
        and event.status is status
        for event in sink.events
    ):
        sink.changed.clear()
        await asyncio.wait_for(sink.changed.wait(), timeout=1)


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
