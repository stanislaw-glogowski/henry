from dataclasses import dataclass
from enum import IntEnum

type AppEvent = AppInitialized | PipelineStageChanged
type AppEventBatch = tuple[AppEvent, ...]


class PipelineStage(IntEnum):
    CAPTURE = 1
    RECORDING = 2
    TRANSCRIPTION = 3
    CONVERSATION = 4
    SYNTHESIS = 5
    PLAYBACK = 6

    @property
    def label(self) -> str:
        match self:
            case PipelineStage.CAPTURE:
                return "Capturing"
            case PipelineStage.RECORDING:
                return "Recording"
            case PipelineStage.TRANSCRIPTION:
                return "Transcribing"
            case PipelineStage.CONVERSATION:
                return "Processing"
            case PipelineStage.SYNTHESIS:
                return "Synthesising"
            case PipelineStage.PLAYBACK:
                return "Playback"


class PipelineStageStatus(IntEnum):
    NONE = 0
    READY = 1
    STARTED = 2
    COMPLETED = 3
    FAILED = 4


@dataclass(frozen=True, slots=True)
class AppInitialized:
    profile_name: str
    vad_threshold: float


@dataclass(frozen=True, slots=True)
class PipelineStageChanged:
    stage: PipelineStage
    status: PipelineStageStatus
