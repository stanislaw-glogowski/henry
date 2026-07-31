from .config import TTSProfile, TTSSettings
from .ports import TTSModel
from .service import SynthesisService


def get_tts_model(
    profile: TTSProfile,
    settings: TTSSettings,
) -> TTSModel:
    from .adapters import get_tts_model as create_tts_model

    return create_tts_model(profile, settings)


__all__ = [
    "get_tts_model",
    "SynthesisService",
    "TTSModel",
    "TTSProfile",
    "TTSSettings",
]
