from dataclasses import dataclass, field, replace

from henry_speech.events import (
    AppEvent,
    PipelineStage,
    PipelineStageChanged,
    PipelineStageStatus,
)

from ..events import TelemetrySnapshot


@dataclass(frozen=True, slots=True)
class PipelineState:
    stages: dict[PipelineStage, PipelineStageStatus] = field(
        default_factory=lambda: {
            stage: PipelineStageStatus.UNKNOWN for stage in PipelineStage
        }
    )


@dataclass(frozen=True)
class State:
    telemetry: TelemetrySnapshot = field(default_factory=TelemetrySnapshot)
    pipeline: PipelineState = field(default_factory=PipelineState)

    def replace_telemetry(self, snapshot: TelemetrySnapshot):
        return replace(
            self,
            telemetry=snapshot,
        )

    def reduce_events(self, event: AppEvent) -> State:
        state = self

        match event:
            case PipelineStageChanged(stage, status):
                pipeline = replace(
                    state.pipeline,
                    stages=state.pipeline.stages | {stage: status},
                )
                state = replace(
                    state,
                    pipeline=pipeline,
                )

        return state
