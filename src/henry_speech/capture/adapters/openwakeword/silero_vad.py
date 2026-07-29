from openwakeword import VAD

from henry_common import PathLocator

from ....audio import AudioFrame
from ...ports import VADModel
from .common import MODELS_PATH


class SileroVADModel(VADModel):
    _MODEL_PATH: str = "silero_vad.onnx"
    _FRAME_SIZE: int = 512

    def __init__(
        self,
        locator: PathLocator,
    ) -> None:
        super().__init__("ONNX")
        self._model: VAD | None = None
        self._model_path = locator.ensure_model_path(MODELS_PATH, self._MODEL_PATH)

    def predict(self, frame: AudioFrame) -> float:
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        probability = self._model.predict(frame.samples, self._FRAME_SIZE)

        return float(probability.item())

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_states()

    def open(self) -> None:
        if self._model is not None:
            raise RuntimeError("Model is already loaded")

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
