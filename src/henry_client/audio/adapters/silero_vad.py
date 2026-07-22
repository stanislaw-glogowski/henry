from types import TracebackType
from typing import Self

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

    def __enter__(self) -> Self:
        self._open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close()

    def predict(self, frame: AudioFrame) -> float:
        model = self._require_model()

        probability, self._state = model.feed(
            frame.samples,
            self._state,
            sample_rate=frame.sample_rate,
        )

        return float(probability.item())

    def _open(self) -> None:
        if self._model is not None:
            raise SileroVADError("Model is already loaded")

        with disable_progress_bars():
            self._logger.debug("Loading model: model_id='{}'", self._MODEL_ID)
            model = load(self._MODEL_ID)
            assert isinstance(model, Model)
            self._model = model

        self._logger.debug("Model READY")

    def _close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._state = None
        self._logger.debug("Model CLOSED")

    def _require_model(self) -> Model:
        if self._model is None:
            raise SileroVADError("Model is not loaded")

        return self._model
