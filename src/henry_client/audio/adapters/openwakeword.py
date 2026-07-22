from types import TracebackType
from typing import Self

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from openwakeword.model import Model

from henry_resources import ensure_model_path

from ..domain import AudioFrame
from ..ports import WakeWordModel


class OpenWakeWordModelError(RuntimeError): ...


class OpenWakeWordModel(WakeWordModel):
    _MODELS_DIR = "openwakeword"
    _MELSPEC_MODEL_PATH = "melspectrogram.onnx"
    _EMBEDDING_MODEL_PATH = "embedding_model.onnx"
    _NUM_CPU = 1
    _FRAME_SIZE = 1280

    def __init__(self, model_path: str) -> None:
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

    def predict(self, frame: AudioFrame) -> float:
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
            raise OpenWakeWordModelError("Model is already loaded")

        self._logger.debug("Loading model: model_path='{}'", self._model_path)

        melspec_path = ensure_model_path(self._MODELS_DIR, self._MELSPEC_MODEL_PATH)
        embedding_path = ensure_model_path(self._MODELS_DIR, self._EMBEDDING_MODEL_PATH)
        wakeword_path = ensure_model_path(self._MODELS_DIR, self._model_path)

        self._model = Model(
            inference_framework="onnx",
            wakeword_models=[
                str(wakeword_path),
            ],
            **{
                "melspec_model_path": str(melspec_path),
                "embedding_model_path": str(embedding_path),
                "ncpu": self._NUM_CPU,
                "device": "cpu",
            },
        )

        self._logger.debug("Model READY")

    def _close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._samples_buffer = np.empty(0, dtype=np.float32)
        self._logger.debug("Model CLOSED")

    def _require_model(self) -> Model:
        if self._model is None:
            raise OpenWakeWordModelError("Model is not loaded")

        return self._model
