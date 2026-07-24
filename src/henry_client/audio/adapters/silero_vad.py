from huggingface_hub.utils import disable_progress_bars
from loguru import logger
from mlx_audio.vad import load
from mlx_audio.vad.models.silero_vad import Model
from mlx_audio.vad.models.silero_vad import SileroVADState as ModelState

from ..domain import AudioFrame
from ..ports import VADModel


class SileroVADError(RuntimeError): ...


class SileroVADModel(VADModel):
    _MODEL_ID = "mlx-community/silero-vad"

    def __init__(self) -> None:
        self._model: Model | None = None
        self._state: ModelState | None = None
        self._logger = logger.bind(component="SileroVADModel")

    def predict(self, frame: AudioFrame) -> float:
        if self._model is None:
            raise SileroVADError("Model is not loaded")

        probability, self._state = self._model.feed(
            frame.samples,
            self._state,
            sample_rate=frame.format.sample_rate,
        )

        return float(probability.item())

    def open(self) -> None:
        if self._model is not None:
            raise SileroVADError("Model is already loaded")

        with disable_progress_bars():
            self._logger.debug("Loading model: model_id='{}'", self._MODEL_ID)
            model = load(self._MODEL_ID)
            assert isinstance(model, Model)
            self._model = model

        self._logger.debug("Model READY")

    def close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._state = None
        self._logger.debug("Model CLOSED")
