import mlx.core as mx
from huggingface_hub.utils import disable_progress_bars
from loguru import logger
from mlx_audio.stt.models.nemo.alignment import AlignedResult
from mlx_audio.stt.models.parakeet import Model
from mlx_audio.stt.utils import load_model

from ...audio import AudioFrame
from ..ports import STTModel


class ParakeetSTTError(RuntimeError): ...


class ParakeetSTTModel(STTModel):
    _MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"

    def __init__(self) -> None:
        self._model: Model | None = None
        self._logger = logger.bind(component="ParakeetSTTModel")

    def transcribe(self, frame: AudioFrame) -> str | None:
        if self._model is None:
            raise ParakeetSTTError("Model is not loaded")

        result = self._model.generate(mx.array(frame.samples))

        if not isinstance(result, AlignedResult):
            return None

        text = result.text.strip()

        if len(text) == 0:
            return None

        return text

    def open(self) -> None:
        if self._model is not None:
            raise ParakeetSTTError("Model is already loaded")

        with disable_progress_bars():
            self._logger.debug("Loading model: model_id='{}'", self._MODEL_ID)
            model = load_model(self._MODEL_ID)
            assert isinstance(model, Model)
            self._model = model

        self._logger.debug("Model READY")

    def close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._logger.debug("Model CLOSED")
