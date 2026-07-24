import asyncio

from loguru import logger
from textual.widgets import Label, RichLog

from henry_cli.events import EventBridge
from henry_cli.logs import LogBuffer
from henry_cli.ui import Layout
from henry_cli.ui.widgets import PipelinePanel, StatusModal, TelemetryPanel
from henry_client.events import AudioCaptured, PipelineStageChanged
from henry_client.pipeline import PipelineStage, PipelineStageStatus


def test_layout_updates_panels_and_handles_actions() -> None:
    async def scenario() -> None:
        events = EventBridge()
        logs = LogBuffer()
        layout = Layout(logs=logs, events=events)

        async with layout.run_test(size=(120, 40)) as pilot:
            pipeline = layout.query_one("#pipeline", PipelinePanel)
            telemetry = layout.query_one("#telemetry", TelemetryPanel)
            log_widget = layout.query_one("#logs", RichLog)

            layout.action_toggle_pipelines()
            layout.action_toggle_telemetry()
            layout.action_toggle_logs()

            assert not pipeline.display
            assert not telemetry.display
            assert not log_widget.display

            events.publish(
                PipelineStageChanged(
                    PipelineStage.RECORDING,
                    PipelineStageStatus.STARTED,
                )
            )
            await pilot.pause()

            assert (
                layout.state.pipeline.stages[PipelineStage.RECORDING]
                is PipelineStageStatus.STARTED
            )

            layout.action_show_status()
            await pilot.pause()

            assert isinstance(layout.screen, StatusModal)
            assert str(layout.screen.query_one("#status-label", Label).render()) == (
                "Recording"
            )

            events.publish(
                PipelineStageChanged(
                    PipelineStage.RECORDING,
                    PipelineStageStatus.COMPLETED,
                )
            )
            await pilot.pause()

            assert str(layout.screen.query_one("#status-label", Label).render()) == (
                "Please wait"
            )

            layout.screen.action_close()
            await pilot.pause()

            logs.write("pipeline message")
            layout.flush_logs()
            assert len(log_widget.lines) == 1

            events.publish(
                AudioCaptured(
                    samples_count=512,
                    speech_score=0.8,
                    speech_detected=True,
                    wakeword_score=0.9,
                    wakeword_detected=True,
                )
            )
            layout.refresh_telemetry()

            assert layout.state.telemetry.captured_sample_count == 512
            assert layout.state.telemetry.wakeword_detected

    try:
        asyncio.run(scenario())
    finally:
        logger.remove()
