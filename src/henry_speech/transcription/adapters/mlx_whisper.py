from collections.abc import Iterator
from typing import Any

from ...audio import AudioFrame
from ..config import MLXWhisperProfile, MLXWhisperSettings
from ..domain import TranscriptionChunk
from .mlx_base import MLXBaseModel


class WhisperModel(MLXBaseModel[Any, MLXWhisperProfile, MLXWhisperSettings]):
    _MODEL_LABEL = "Whisper"

    def transcribe(self, frame: AudioFrame) -> Iterator[TranscriptionChunk]:
        model = self._require_model()

        # TODO: add supported options from profile
        options: dict[str, Any] = {
            "task": "transcribe",
            "verbose": None,
        }
        if self._profile.language is not None:
            options["language"] = self._profile.language

        result = model.generate(
            frame.samples,
            **options,
        )
        if text := result.text.strip():
            yield TranscriptionChunk(content=text)
