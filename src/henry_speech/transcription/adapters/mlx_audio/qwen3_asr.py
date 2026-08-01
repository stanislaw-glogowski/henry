from collections.abc import Iterator
from typing import TYPE_CHECKING

from huggingface_hub.utils import disable_progress_bars

from ....audio import AudioFrame
from ...config import STTProfile
from ...domain import TranscriptionChunk
from ...ports import STTModel

if TYPE_CHECKING:
    from mlx_audio.stt.models.qwen3_asr import Model


class Qwen3ASRModel(STTModel):
    """MLX Qwen3-ASR adapter for complete utterances."""

    _DEFAULT_MODEL_ID = "mlx-community/Qwen3-ASR-0.6B-8bit"

    def __init__(self, profile: STTProfile | None) -> None:
        super().__init__("MLX")
        self._profile = profile if profile is not None else STTProfile()
        self._model: Model | None = None

    def transcribe(self, frame: AudioFrame) -> Iterator[TranscriptionChunk]:
        if self._model is None:
            raise RuntimeError("Qwen3-ASR model is not loaded")

        result = self._model.generate(
            frame.samples,
            language="Polish",
            temperature=0.0,
        )
        if text := result.text.strip():
            yield TranscriptionChunk(content=text)

    def open(self) -> None:
        if self._model is not None:
            raise RuntimeError("Qwen3-ASR model is already loaded")

        model_id = (
            self._profile.model if self._profile.model else self._DEFAULT_MODEL_ID
        )

        with disable_progress_bars():
            from mlx_audio.stt.models.qwen3_asr import Model as LoadedModel
            from mlx_audio.stt.utils import load_model

            self._logger.debug("Loading model: model_id='{}'", model_id)
            model = load_model(model_id)
            assert isinstance(model, LoadedModel)
            self._model = model

        self._logger.debug("Model READY")

    def close(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._logger.debug("Model CLOSED")
