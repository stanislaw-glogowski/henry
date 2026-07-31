from .config import STTProfile, STTSettings
from .domain import Transcription, TranscriptionChunk, TranscriptionText
from .ports import STTModel
from .service import TranscriptionService


def get_stt_model(
    profile: STTProfile,
    settings: STTSettings,
) -> STTModel:
    from .adapters import get_stt_model as create_stt_model

    return create_stt_model(profile, settings)


__all__ = [
    "STTModel",
    "STTProfile",
    "STTSettings",
    "Transcription",
    "TranscriptionChunk",
    "TranscriptionService",
    "TranscriptionText",
    "get_stt_model",
]
