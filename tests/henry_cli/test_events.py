import asyncio

from henry_cli.events import ReplyDraft, UiEventBridge
from henry_common.events import EventBus, ShutdownEvent
from henry_conversation import (
    CancelReply,
    ConversationReady,
    ReplyDraftUpdated,
    ReplyGenerationCompleted,
)
from henry_speech.events import (
    InteractionTimingObserved,
    SpeechChunkCaptured,
    TranscriptionProgressObserved,
    UserTurnCommitted,
    VADObserved,
    WakeWordObserved,
)


def test_ui_event_bridge_coalesces_telemetry_and_queues_state_events() -> None:
    async def scenario() -> None:
        bus = EventBus()
        bridge = UiEventBridge()
        task = asyncio.create_task(bridge.run(bus))
        await asyncio.wait_for(bridge.wait_ready(), 1)

        bus.publish(
            SpeechChunkCaptured(160, True, False),
            SpeechChunkCaptured(80, False, False),
            VADObserved(0.8, True),
            WakeWordObserved(0.7, False),
            TranscriptionProgressObserved(3, "Hello", True),
            ReplyDraftUpdated(7, "Draft"),
            InteractionTimingObserved("reply_started", 123.4),
        )
        await asyncio.sleep(0)
        snapshot = bridge.telemetry_snapshot
        assert snapshot.captured_sample_count == 240
        assert snapshot.vad_score == 0.8
        assert snapshot.vad_detected
        assert snapshot.wakeword_score == 0.7
        assert snapshot.transcription is not None
        assert snapshot.transcription.turn_id == 3
        assert snapshot.reply is not None
        assert snapshot.reply.text == "Draft"
        assert snapshot.timings == (("reply_started", 123.4),)

        reliable = ConversationReady()
        bus.publish(reliable)
        assert await asyncio.wait_for(bridge.receive(), 1) == reliable
        bridge.task_done()

        bus.publish(UserTurnCommitted(3, "Hello"))
        assert isinstance(
            await asyncio.wait_for(bridge.receive(), 1), UserTurnCommitted
        )
        bridge.task_done()
        assert bridge.telemetry_snapshot.transcription is None

        bus.publish(ReplyDraftUpdated(7, ""))
        await asyncio.sleep(0)
        assert bridge.telemetry_snapshot.reply is None
        bus.publish(ReplyDraftUpdated(7, "again"), ReplyGenerationCompleted(7))
        event = await asyncio.wait_for(bridge.receive(), 1)
        bridge.task_done()
        assert event == ReplyGenerationCompleted(7)
        assert bridge.telemetry_snapshot.reply is None

        bus.publish(ReplyDraftUpdated(8, "again"), CancelReply("", 8))
        event = await asyncio.wait_for(bridge.receive(), 1)
        bridge.task_done()
        assert event == ReplyDraftUpdated(8, "again")
        event = await asyncio.wait_for(bridge.receive(), 1)
        bridge.task_done()
        assert event == CancelReply("", 8)
        assert bridge.telemetry_snapshot.reply is None

        bus.publish(ReplyDraftUpdated(9, "new reply"), CancelReply("", 8))
        event = await asyncio.wait_for(bridge.receive(), 1)
        bridge.task_done()
        assert event == CancelReply("", 8)
        assert bridge.telemetry_snapshot.reply == ReplyDraft(9, "new reply")
        bus.publish(ReplyGenerationCompleted(8))
        event = await asyncio.wait_for(bridge.receive(), 1)
        bridge.task_done()
        assert event == ReplyGenerationCompleted(8)
        assert bridge.telemetry_snapshot.reply == ReplyDraft(9, "new reply")

        bus.publish(ShutdownEvent())
        assert isinstance(await asyncio.wait_for(bridge.receive(), 1), ShutdownEvent)
        bridge.task_done()
        await asyncio.wait_for(task, 1)

    asyncio.run(scenario())


def test_ui_event_bridge_rejects_reliable_queue_overflow() -> None:
    bridge = UiEventBridge()
    for _ in range(bridge._QUEUE_MAXSIZE):
        bridge._queue.put_nowait(ConversationReady())
    try:
        bridge._queue.put_nowait(ConversationReady())
    except asyncio.QueueFull:
        pass
    else:
        raise AssertionError("expected reliable UI queue to be bounded")
