import asyncio

from henry_client.events import AudioPlayed, PipelineStageChanged
from henry_client.orchestrator import ListeningMode, Orchestrator
from henry_client.pipeline import PipelineStage, PipelineStageStatus
from tests.support import (
    FakeAudioService,
    FakeConversationService,
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


def test_orchestrator_plays_preloaded_wakeword_reply_then_records() -> None:
    async def scenario() -> None:
        audio = FakeAudioService()
        speech = FakeSpeechService()
        events = RecordingEventSink()
        shutdown = asyncio.Event()
        orchestrator = Orchestrator(
            audio=audio,
            speech=speech,
            conversation=FakeConversationService("answer"),
            events=events,
            wakeword_reply_text="Ready.",
        )
        orchestrator._WAKEWORD_REPLY_START_DELAY_SECONDS = 0
        orchestrator._WAKEWORD_REPLY_END_DELAY_SECONDS = 0

        task = asyncio.create_task(orchestrator.run(shutdown))
        await wait_for_event(
            events,
            PipelineStage.LISTENING,
            PipelineStageStatus.STARTED,
        )
        await audio.chunks.put(
            chunk(
                speech_detected=True,
                speech_score=0.9,
                wakeword_detected=True,
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

    asyncio.run(scenario())


def test_orchestrator_runs_utterance_through_full_reply_pipeline() -> None:
    async def scenario() -> None:
        audio = FakeAudioService()
        speech = FakeSpeechService(transcript="question")
        conversation = FakeConversationService("First answer.", "Second answer.")
        events = RecordingEventSink()
        shutdown = asyncio.Event()
        orchestrator = Orchestrator(
            audio=audio,
            speech=speech,
            conversation=conversation,
            events=events,
        )
        orchestrator._WAKEWORD_REPLY_START_DELAY_SECONDS = 0
        orchestrator._WAKEWORD_REPLY_END_DELAY_SECONDS = 0

        task = asyncio.create_task(orchestrator.run(shutdown))
        await wait_for_event(
            events,
            PipelineStage.LISTENING,
            PipelineStageStatus.STARTED,
        )
        await audio.chunks.put(chunk(wakeword_detected=True, wakeword_score=0.9))
        await wait_for_event(
            events,
            PipelineStage.RECORDING,
            PipelineStageStatus.STARTED,
        )
        await audio.chunks.put(chunk(speech_detected=True, speech_score=0.9))
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

        assert conversation.received == ["question"]
        assert speech.synthesized == ["First answer.", "Second answer."]
        assert len(audio.written) == 2

    asyncio.run(scenario())
