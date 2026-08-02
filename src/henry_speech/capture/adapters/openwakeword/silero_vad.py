from openwakeword import VAD

from henry_resources.models import ModelCatalog

from ....audio import AudioFrame
from ...config import VADSettings
from ...domain import DetectionResult
from ...ports import VADModel
from .base import BaseModel


class SileroVADModel(VADModel, BaseModel):
    _MODEL_PATH: str = "silero_vad.onnx"
    _FRAME_SIZE: int = 512

    def __init__(
        self,
        catalog: ModelCatalog,
        settings: VADSettings | None = None,
    ) -> None:
        super().__init__(catalog=catalog, context="ONNX")
        self._settings = settings if settings is not None else VADSettings()
        self._model: VAD | None = None
        self._model_path = self._ensure_model_path(self._MODEL_PATH)

    def detect(self, frame: AudioFrame) -> DetectionResult:
        if self._model is None:
            raise RuntimeError("OpenWakeWord Silero VAD model is not loaded")

        probability = self._model.predict(frame.samples, self._FRAME_SIZE)

        score = float(probability.item())

        return DetectionResult(
            score=score,
            detected=score > self._settings.threshold,
        )

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_states()

    def open(self) -> None:
        if self._model is not None:
            raise RuntimeError("OpenWakeWord Silero VAD model is already loaded")

        self._logger.debug("Loading model: model_path='{}'", self._model_path.name)

        self._model = VAD(
            model_path=str(self._model_path),
        )

        self._logger.debug("Model READY")

    def close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._logger.debug("Model CLOSED")
