from collections.abc import Iterator

import mlx.core as mx
from huggingface_hub.utils import disable_progress_bars
from mlx_audio.stt.models.parakeet import Model
from mlx_audio.stt.utils import load_model

from ....audio import AudioFrame
from ...domain import TranscriptionChunk
from ...ports import TranscriptionModel


class ParakeetTDTModel(TranscriptionModel):
    _MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"

    def __init__(self) -> None:
        super().__init__("MLX")
        self._model: Model | None = None

    def transcribe(self, frame: AudioFrame) -> Iterator[TranscriptionChunk]:
        if self._model is None:
            raise RuntimeError("Model is not loaded")

        for chunk in self._model.stream_generate(mx.array(frame.samples)):
            yield TranscriptionChunk(
                content=chunk.text,
            )

    def open(self) -> None:
        if self._model is not None:
            raise RuntimeError("Model is already loaded")

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
