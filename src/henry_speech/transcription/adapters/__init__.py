from typing import TYPE_CHECKING

from ..config import (
    MLXParakeetTDTSettings,
    MLXQwen3ASRSettings,
    MLXWhisperSettings,
)

if TYPE_CHECKING:
    from ..config import STTProfile, STTSettings
    from ..ports import STTModel


def get_stt_model(
    profile: STTProfile,
    settings: STTSettings,
) -> STTModel:
    match settings:
        case MLXParakeetTDTSettings():
            from .mlx_parakeet_tdt import ParakeetTDTModel

            return ParakeetTDTModel(profile.stt_mlx_parakeet_tdt, settings)
        case MLXQwen3ASRSettings():
            from .mlx_qwen3_asr import Qwen3ASRModel

            return Qwen3ASRModel(profile.stt_mlx_qwen3_asr, settings)
        case MLXWhisperSettings():
            from .mlx_whisper import WhisperModel

            return WhisperModel(profile.stt_mlx_whisper, settings)
        case _:
            raise ValueError(f"Unsupported STT settings: {settings!r}")
