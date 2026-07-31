from ..config import STTProfile, STTSettings
from ..ports import STTModel


def get_stt_model(
    profile: STTProfile,
    settings: STTSettings,
) -> STTModel:
    match settings.adapter:
        case "mlx:parakeet-tdt":
            from .mlx_audio import ParakeetTDTModel

            return ParakeetTDTModel(profile)
        case _:
            raise ValueError(f"Unknown STT adapter: {settings.adapter}")
