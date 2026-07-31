from collections.abc import Iterator

import mlx.core as mx
from huggingface_hub.utils import disable_progress_bars
from mlx_audio.stt.models.parakeet import Model
from mlx_audio.stt.utils import load_model

from ....audio import AudioFrame
from ...config import STTProfile
from ...domain import TranscriptionChunk
from ...ports import STTModel


class ParakeetTDTModel(STTModel):
    _DEFAULT_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"

    def __init__(self, profile: STTProfile | None) -> None:
        super().__init__("MLX")
        self._profile = profile if profile is not None else STTProfile()
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

        model_id = (
            self._profile.model if self._profile.model else self._DEFAULT_MODEL_ID
        )

        with disable_progress_bars():
            self._logger.debug("Loading model: model_id='{}'", model_id)
            model = load_model(model_id)
            assert isinstance(model, Model)
            self._model = model

        self._logger.debug("Model READY")

    def close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._logger.debug("Model CLOSED")
