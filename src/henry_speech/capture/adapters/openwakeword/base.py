from abc import ABC
from pathlib import Path

from henry_resources.models import ModelCatalog
from henry_speech.capture.ports import DetectionModel


class BaseModel(DetectionModel, ABC):
    _MODELS_PATH = Path("openwakeword")

    def __init__(self, catalog: ModelCatalog, context: str | None = None):
        self._catalog = catalog
        super().__init__(context)

    def _ensure_model_path(self, model_path: str) -> Path:
        if not model_path.endswith(".onnx"):
            raise ValueError(f"OpenWakeWord model must be an ONNX file: {model_path!r}")

        return self._catalog.ensure_model_path(self._MODELS_PATH, model_path)
