import asyncio

from henry_cli.events import EventBridge
from henry_client.events import AudioCaptured, PipelineStageChanged
from henry_client.pipeline import PipelineStage, PipelineStageStatus


def test_event_bridge_coalesces_telemetry_and_resets_disabled_wakeword() -> None:
    bridge = EventBridge()
    bridge.publish(
        AudioCaptured(
            samples_count=512,
            speech_score=0.7,
            speech_detected=True,
            wakeword_score=0.9,
            wakeword_detected=True,
        ),
        AudioCaptured(
            samples_count=512,
            speech_score=0.1,
            speech_detected=False,
            wakeword_score=None,
            wakeword_detected=None,
        ),
    )

    snapshot = bridge.telemetry_snapshot
    assert snapshot.captured_sample_count == 1_024
    assert snapshot.wakeword_score == 0.0
    assert snapshot.wakeword_detected is False


def test_event_bridge_queues_reliable_events() -> None:
    async def scenario() -> None:
        bridge = EventBridge()
        event = PipelineStageChanged(
            PipelineStage.CAPTURE,
            PipelineStageStatus.STARTED,
        )
        bridge.publish(event)

        assert await asyncio.wait_for(bridge.receive(), timeout=1) is event

    asyncio.run(scenario())
