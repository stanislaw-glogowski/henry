from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import STTProfile, STTSettings
    from ..ports import STTModel


def get_stt_model(
    profile: STTProfile,
    settings: STTSettings,
) -> STTModel:
    match settings.adapter:
        case "mlx:parakeet-tdt":
            from .mlx_audio.parakeet_tdt import ParakeetTDTModel

            return ParakeetTDTModel(profile)
        case "mlx:qwen3-asr":
            from .mlx_audio.qwen3_asr import Qwen3ASRModel

            return Qwen3ASRModel(profile)
        case "mlx:whisper":
            from .mlx_audio.whisper import WhisperModel

            return WhisperModel(profile)
        case _:
            raise ValueError(f"Unsupported STT adapter: {settings.adapter!r}")
