from collections.abc import Iterator
from typing import Any

import numpy as np

from ...audio import AudioFormat, AudioFrame
from ..config import MLXChatterboxProfile, MLXChatterboxSettings
from .mlx_base import MLXBaseModel


class MLXChatterboxModel(MLXBaseModel[MLXChatterboxProfile, MLXChatterboxSettings]):
    """Multilingual Chatterbox adapter backed by MLX-Audio."""

    _MODEL_LABEL = "MLX Chatterbox"

    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        model = self._require_model()

        # TODO: add supported options from profile and settings
        options: dict[str, Any] = {
            "verbose": False,
            "lang_code": self._profile.lang_code or self._settings.lang_code,
        }

        for result in model.generate(text=text, **options):
            yield AudioFrame(
                samples=np.ascontiguousarray(result.audio, dtype=np.float32),
                format=AudioFormat(sample_rate=result.sample_rate, channels=1),
            )
