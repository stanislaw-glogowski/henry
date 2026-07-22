from types import TracebackType
from typing import Self

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from openwakeword.model import Model

from henry_resources import ensure_model_path

from ...audio import AudioChunk
from ..ports import WakeWordModel

MODELS_DIR = "openwakeword"
MELSPEC_MODEL_PATH = "melspectrogram.onnx"
EMBEDDING_MODEL_PATH = "embedding_model.onnx"


class SileroWakeWordError(RuntimeError): ...


class OpenWakeWordModel(WakeWordModel):
    _FRAME_SIZE = 1280

    def __init__(self, model_path: str = "Hey_Henree_20260406_162745.onnx") -> None:
        self._model_path = model_path
        self._model: Model | None = None
        self._samples_buffer = np.empty(0, dtype=np.float32)
        self._logger = logger.bind(component="OpenWakeWordModel")

    def __enter__(self) -> Self:
        self._open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close()

    def predict(self, frame: AudioChunk) -> float:
        model = self._require_model()

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
            return 0.0

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

            scores = model.predict(pcm16_chunk)

            for score in scores.values():
                highest_score = max(highest_score, float(score))

        return highest_score

    def reset(self) -> None:
        self._samples_buffer = np.empty(0, dtype=np.float32)

        if self._model is not None:
            self._model.reset()

    def _open(self) -> None:
        if self._model is not None:
            raise SileroWakeWordError("Model is already loaded")

        self._logger.debug("Loading model: model_path='{}'", self._model_path)

        feature_options = {
            "melspec_model_path": str(
                ensure_model_path(MODELS_DIR, "melspectrogram.onnx")
            ),
            "embedding_model_path": str(
                ensure_model_path(MODELS_DIR, "embedding_model.onnx")
            ),
            "ncpu": 1,
            "device": "cpu",
        }

        self._model = Model(
            inference_framework="onnx",
            wakeword_models=[
                str(ensure_model_path(MODELS_DIR, self._model_path)),
            ],
            **feature_options,
        )

        self._logger.debug("Model READY")

    def _close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._state = None
        self._logger.debug("Model CLOSED")

    def _require_model(self) -> Model:
        if self._model is None:
            raise SileroWakeWordError("Model is not loaded")

        return self._model
