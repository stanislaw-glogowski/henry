from enum import Enum, auto


class PipelineStage(Enum):
    CAPTURE = auto()
    LISTENING = auto()
    RECORDING = auto()
    TRANSCRIPTION = auto()
    PROCESSING = auto()
    SYNTHESIS = auto()
    PLAYBACK = auto()

    @property
    def label(self) -> str:
        match self:
            case PipelineStage.CAPTURE:
                return "Capturing"
            case PipelineStage.LISTENING:
                return "Listening"
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


class PipelineStageStatus(Enum):
    UNKNOWN = auto()
    READY = auto()
    STARTED = auto()
    COMPLETED = auto()
    FAILED = auto()
