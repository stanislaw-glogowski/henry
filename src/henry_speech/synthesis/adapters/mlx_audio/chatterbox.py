from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub.utils import disable_progress_bars

from ....audio import AudioFormat, AudioFrame
from ...config import TTSProfile
from ...ports import TTSModel

if TYPE_CHECKING:
    from mlx_audio.tts.models.chatterbox import Model


class ChatterboxTTSModel(TTSModel):
    """Multilingual Chatterbox adapter backed by MLX-Audio."""

    _DEFAULT_MODEL_ID = "mlx-community/chatterbox-fp16"

    def __init__(self, profile: TTSProfile) -> None:
        super().__init__("MLX")
        self._profile = profile
        self._model: Model | None = None

    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        if self._model is None:
            raise RuntimeError("Chatterbox model is not loaded")

        for result in self._model.generate(
            text=text,
            lang_code="pl",
            verbose=False,
        ):
            yield AudioFrame(
                samples=np.ascontiguousarray(result.audio, dtype=np.float32),
                format=AudioFormat(sample_rate=result.sample_rate, channels=1),
            )

    def open(self) -> None:
        if self._model is not None:
            raise RuntimeError("Chatterbox model is already loaded")

        with disable_progress_bars():
            from mlx_audio.tts.models.chatterbox import Model as LoadedModel
            from mlx_audio.tts.utils import load

            model_id = self._profile.model or self._DEFAULT_MODEL_ID
            self._logger.debug("Loading model: model_id='{}'", model_id)
            model = load(model_id)
            assert isinstance(model, LoadedModel)
            self._model = model

        self._logger.debug("Model READY")

    def close(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._logger.debug("Model CLOSED")
