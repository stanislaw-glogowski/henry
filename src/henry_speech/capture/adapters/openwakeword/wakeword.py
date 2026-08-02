import numpy as np
from numpy.typing import NDArray
from openwakeword import Model

from henry_resources.models import ModelCatalog

from ....audio import AudioFrame
from ...config import WakeWordProfile
from ...domain import DetectionResult
from ...ports import WakeWordModel
from .base import BaseModel


class OpenWakeWordModel(WakeWordModel, BaseModel):
    _MELSPEC_PATH = "melspectrogram.onnx"
    _EMBEDDING_PATH = "embedding_model.onnx"
    _NUM_CPU = 1
    _FRAME_SIZE = 1280

    def __init__(
        self,
        catalog: ModelCatalog,
        profile: WakeWordProfile,
    ) -> None:
        super().__init__(catalog)
        self._profile = profile
        self._model: Model | None = None
        self._model_path = self._ensure_model_path(profile.model_path)
        self._melspec_path = self._ensure_model_path(self._MELSPEC_PATH)
        self._embedding_path = self._ensure_model_path(self._EMBEDDING_PATH)
        self._samples_buffer = np.empty(0, dtype=np.float32)

    def detect(self, frame: AudioFrame) -> DetectionResult:
        if self._model is None:
            raise RuntimeError("OpenWakeWord model is not loaded")

        incoming_samples: NDArray[np.float32] = np.asarray(
            frame.samples,
            dtype=np.float32,
        ).reshape(-1)

        samples = np.asarray(
            np.concatenate(
                (self._samples_buffer, incoming_samples),
            ),
            dtype=np.float32,
        )

        complete_samples = (samples.size // self._FRAME_SIZE) * self._FRAME_SIZE

        if complete_samples == 0:
            self._samples_buffer = samples
            return DetectionResult()

        self._samples_buffer = samples[complete_samples:].copy()

        chunks: NDArray[np.float32] = samples[:complete_samples].reshape(
            -1,
            self._FRAME_SIZE,
        )

        highest_score = 0.0

        for chunk in chunks:
            float_chunk = np.asarray(
                chunk,
                dtype=np.float32,
            )

            clipped = np.asarray(
                np.clip(float_chunk, -1.0, 1.0),
                dtype=np.float32,
            )

            scaled = clipped * np.float32(32767.0)

            pcm16_chunk = np.asarray(
                np.rint(scaled),
                dtype=np.int16,
            )

            scores = self._model.predict(pcm16_chunk)
            if not isinstance(scores, dict):
                raise RuntimeError("OpenWakeWord returned timing data unexpectedly")

            for score in scores.values():
                highest_score = max(highest_score, float(score))

        return DetectionResult(
            score=highest_score,
            detected=highest_score > self._profile.threshold,
        )

    def reset(self) -> None:
        self._samples_buffer = np.empty(0, dtype=np.float32)

        if self._model is not None:
            self._model.reset()

    def open(self) -> None:
        if self._model is not None:
            raise RuntimeError("OpenWakeWord model is already loaded")

        self._logger.debug("Loading model: model_path='{}'", self._model_path.name)

        self._model = Model(
            inference_framework="onnx",
            wakeword_models=[
                str(self._model_path),
            ],
            **{
                "melspec_model_path": str(self._melspec_path),
                "embedding_model_path": str(self._embedding_path),
                "ncpu": self._NUM_CPU,
                "device": "cpu",
            },
        )

        self._logger.debug("Model READY")

    def close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._samples_buffer = np.empty(0, dtype=np.float32)
        self._logger.debug("Model CLOSED")
