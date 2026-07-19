from dataclasses import dataclass, field, replace

from henry.events import (
    AppEventBatch,
    AppInitialized,
    PipelineStage,
    PipelineStageChanged,
    PipelineStageStatus,
)

from ..telemetry import TelemetrySnapshot


@dataclass(frozen=True, slots=True)
class PipelineState:
    stages: dict[PipelineStage, PipelineStageStatus] = field(
        default_factory=lambda: {
            stage: PipelineStageStatus.NONE for stage in PipelineStage
        }
    )


@dataclass(frozen=True)
class State:
    info: AppInitialized | None = None
    telemetry: TelemetrySnapshot = field(default_factory=TelemetrySnapshot)
    pipeline: PipelineState = field(default_factory=PipelineState)

    def replace_telemetry(self, snapshot: TelemetrySnapshot):
        return replace(
            self,
            telemetry=snapshot,
        )

    def reduce_events(self, events: AppEventBatch) -> State:
        state = self

        for event in events:
            match event:
                case AppInitialized() as info:
                    state = replace(state, info=info)
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
