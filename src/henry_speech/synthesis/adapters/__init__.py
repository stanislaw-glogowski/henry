from ..config import TTSProfile, TTSSettings
from ..ports import TTSModel


def get_tts_model(
    profile: TTSProfile,
    settings: TTSSettings,
) -> TTSModel:
    match settings.adapter:
        case "piper":
            from .piper import PiperModel

            return PiperModel(profile)
        case _:
            raise ValueError(f"Unknown TTS adapter: {settings.adapter}")
