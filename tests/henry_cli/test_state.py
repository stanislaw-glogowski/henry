from henry_cli.events import TelemetrySnapshot
from henry_cli.ui.state import PipelineState, State
from henry_client.events import AppEvent, PipelineStageChanged
from henry_client.pipeline import PipelineStage, PipelineStageStatus


def test_pipeline_state_starts_with_every_stage_unknown() -> None:
    state = PipelineState()

    assert set(state.stages) == set(PipelineStage)
    assert set(state.stages.values()) == {PipelineStageStatus.UNKNOWN}


def test_state_reduces_pipeline_event_without_mutating_previous_state() -> None:
    state = State()

    updated = state.reduce_events(
        PipelineStageChanged(
            PipelineStage.RECORDING,
            PipelineStageStatus.STARTED,
        )
    )

    assert state.pipeline.stages[PipelineStage.RECORDING] is PipelineStageStatus.UNKNOWN
    assert (
        updated.pipeline.stages[PipelineStage.RECORDING] is PipelineStageStatus.STARTED
    )
    assert updated.telemetry is state.telemetry


def test_state_ignores_unhandled_event_and_replaces_telemetry() -> None:
    state = State()
    snapshot = TelemetrySnapshot(captured_sample_count=512)

    assert state.reduce_events(AppEvent()) is state
    assert state.replace_telemetry(snapshot).telemetry is snapshot
