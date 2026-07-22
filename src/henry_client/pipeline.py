from enum import IntEnum


class PipelineStage(IntEnum):
    CAPTURE = 1
    RECORDING = 2
    TRANSCRIPTION = 3
    PROCESSING = 4
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
            case PipelineStage.PROCESSING:
                return "Processing"
            case PipelineStage.SYNTHESIS:
                return "Synthesising"
            case PipelineStage.PLAYBACK:
                return "Playback"


class PipelineStageStatus(IntEnum):
    UNKNOWN = 0
    READY = 1
    STARTED = 2
    COMPLETED = 3
    FAILED = -1
