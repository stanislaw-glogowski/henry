from collections.abc import Iterator
from typing import Any

from ...audio import AudioFrame
from ..config import MLXParakeetTDTProfile, MLXParakeetTDTSettings
from ..domain import TranscriptionChunk
from .mlx_base import MLXBaseModel


class ParakeetTDTModel(MLXBaseModel[MLXParakeetTDTProfile, MLXParakeetTDTSettings]):
    _MODEL_LABEL = "Parakeet TDT"

    def transcribe(self, frame: AudioFrame) -> Iterator[TranscriptionChunk]:
        import mlx.core as mx

        model = self._require_model()

        # TODO: add supported options from profile and settings
        options: dict[str, Any] = {}

        for chunk in model.stream_generate(mx.array(frame.samples), **options):
            yield TranscriptionChunk(
                content=chunk.text,
            )
