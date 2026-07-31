from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from henry_resources.models import ModelCatalog

from .config import VADSettings, WakeWordProfile, WakeWordSettings
from .domain import DetectionResult, SpeechChunk
from .ports import VADModel, WakeWordModel
from .service import CaptureService


def get_vad_model(
    catalog: ModelCatalog,
    settings: VADSettings,
) -> VADModel:
    from .adapters import get_vad_model as create_vad_model

    return create_vad_model(catalog, settings)


def get_wakeword_model(
    catalog: ModelCatalog,
    profile: WakeWordProfile,
    settings: WakeWordSettings,
) -> WakeWordModel:
    from .adapters import get_wakeword_model as create_wakeword_model

    return create_wakeword_model(catalog, profile, settings)


__all__ = [
    "CaptureService",
    "DetectionResult",
    "SpeechChunk",
    "VADModel",
    "VADSettings",
    "WakeWordModel",
    "WakeWordProfile",
    "WakeWordSettings",
    "get_vad_model",
    "get_wakeword_model",
]
