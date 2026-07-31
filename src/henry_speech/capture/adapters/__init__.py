from henry_resources.models import ModelCatalog

from ..config import VADSettings, WakeWordProfile, WakeWordSettings
from ..ports import VADModel, WakeWordModel


def get_vad_model(
    catalog: ModelCatalog,
    settings: VADSettings,
) -> VADModel:
    match settings.adapter:
        case "mlx:silero_vad":
            from .mlx_audio import SileroVADModel

            return SileroVADModel(settings)
        case "openwakeword":
            from .openwakeword import SileroVADModel

            return SileroVADModel(catalog, settings)
        case _:
            raise ValueError(f"Unknown VAD adapter: {settings.adapter}")


def get_wakeword_model(
    catalog: ModelCatalog,
    profile: WakeWordProfile,
    settings: WakeWordSettings,
) -> WakeWordModel:
    match settings.adapter:
        case "openwakeword":
            from .openwakeword import OpenWakeWordModel

            return OpenWakeWordModel(catalog, profile)
        case _:
            raise ValueError(f"Unknown wake word adapter: {settings.adapter}")
