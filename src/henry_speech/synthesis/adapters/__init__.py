from typing import TYPE_CHECKING

from ..config import MLXChatterboxSettings, PiperSettings

if TYPE_CHECKING:
    from ..config import TTSProfile, TTSSettings
    from ..ports import TTSModel


def get_tts_model(
    profile: TTSProfile,
    settings: TTSSettings,
) -> TTSModel:
    match settings:
        case PiperSettings():
            from .piper import PiperModel

            return PiperModel(profile.tts_piper, settings)
        case MLXChatterboxSettings():
            from .mlx_chatterbox import MLXChatterboxModel

            return MLXChatterboxModel(profile.tts_mlx_chatterbox, settings)
        case _:
            raise ValueError(f"Unsupported TTS settings: {settings!r}")
