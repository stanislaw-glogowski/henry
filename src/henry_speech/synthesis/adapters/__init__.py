from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import TTSProfile, TTSSettings
    from ..ports import TTSModel


def get_tts_model(
    profile: TTSProfile,
    settings: TTSSettings,
) -> TTSModel:
    match settings.adapter:
        case "piper":
            from .piper import PiperTTSModel

            return PiperTTSModel(profile)
        case "mlx:chatterbox":
            from .mlx_audio.chatterbox import ChatterboxTTSModel

            return ChatterboxTTSModel(profile)
        case _:
            raise ValueError(f"Unsupported TTS adapter: {settings.adapter!r}")
