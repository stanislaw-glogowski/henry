from collections.abc import Iterator
from typing import TYPE_CHECKING

from huggingface_hub.utils import disable_progress_bars

from ....audio import AudioFrame
from ...config import STTProfile
from ...domain import TranscriptionChunk
from ...ports import STTModel

if TYPE_CHECKING:
    from mlx_audio.stt.models.whisper import Model


class WhisperModel(STTModel):
    """MLX Whisper adapter retained as an offline benchmark candidate."""

    _DEFAULT_MODEL_ID = "mlx-community/whisper-large-v3-turbo-asr-fp16"

    def __init__(self, profile: STTProfile | None) -> None:
        super().__init__("MLX")
        self._profile = profile if profile is not None else STTProfile()
        self._model: Model | None = None

    def transcribe(self, frame: AudioFrame) -> Iterator[TranscriptionChunk]:
        if self._model is None:
            raise RuntimeError("Whisper model is not loaded")

        options = {
            "task": "transcribe",
            "verbose": None,
        }
        if self._profile.language is not None:
            options["language"] = self._profile.language

        result = self._model.generate(frame.samples, **options)
        if text := result.text.strip():
            yield TranscriptionChunk(content=text)

    def open(self) -> None:
        if self._model is not None:
            raise RuntimeError("Whisper model is already loaded")

        model_id = (
            self._profile.model if self._profile.model else self._DEFAULT_MODEL_ID
        )

        with disable_progress_bars():
            from mlx_audio.stt.models.whisper import Model as LoadedModel
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
