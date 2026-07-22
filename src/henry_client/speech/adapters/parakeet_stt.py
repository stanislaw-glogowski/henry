from types import TracebackType
from typing import Self

import mlx.core as mx
from huggingface_hub.utils import disable_progress_bars
from loguru import logger
from mlx_audio.stt.models.nemo.alignment import AlignedResult
from mlx_audio.stt.models.parakeet import Model
from mlx_audio.stt.utils import load_model

from ...audio import AudioFrame
from ..ports import STTModel

MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"


class ParakeetSTTError(RuntimeError): ...


class ParakeetSTTModel(STTModel):
    def __init__(self) -> None:
        self._model: Model | None = None
        self._logger = logger.bind(component="ParakeetSTTModel")

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

    def transcribe(self, frame: AudioFrame) -> str | None:
        model = self._require_model()

        result = model.generate(mx.array(frame.samples))

        if not isinstance(result, AlignedResult):
            return None

        text = result.text.strip()

        if len(text) == 0:
            return None

        return text

    def _open(self) -> None:
        if self._model is not None:
            raise ParakeetSTTError("Model is already loaded")

        with disable_progress_bars():
            self._logger.debug("Loading model: model_id='{}'", MODEL_ID)
            model = load_model(MODEL_ID)
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
            raise ParakeetSTTError("Model is not loaded")

        return self._model
