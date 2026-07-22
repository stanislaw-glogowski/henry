from collections.abc import Iterator
from types import TracebackType
from typing import Self

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from loguru import logger
from piper import PiperVoice, SynthesisConfig

from ...audio import AudioFrame
from ..ports import TTSModel

REPOSITORY_ID = "rhasspy/piper-voices"
LENGTH_SCALE = 1.05
NOISE_SCALE = None
NOISE_W_SCALE = None


class PiperTTSError(RuntimeError): ...


class PiperTTSModel(TTSModel):
    def __init__(
        self,
        model_path: str,
        length_scale: float = LENGTH_SCALE,
        noise_scale: float | None = NOISE_SCALE,
        noise_w_scale: float | None = NOISE_W_SCALE,
        repository_id: str = REPOSITORY_ID,
    ) -> None:
        self._length_scale = length_scale
        self._noise_scale = noise_scale
        self._noise_w_scale = noise_w_scale
        self._repository_id = repository_id
        self._model_path = model_path
        self._model: PiperVoice | None = None
        self._logger = logger.bind(component="PiperTTSModel")

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

    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        model = self._require_model()

        chunks = model.synthesize(
            text=text,
            include_alignments=False,
            syn_config=SynthesisConfig(
                length_scale=self._length_scale,
                noise_scale=self._noise_scale,
                noise_w_scale=self._noise_w_scale,
                normalize_audio=True,
            ),
        )

        for chunk in chunks:
            samples = np.ascontiguousarray(
                chunk.audio_float_array,
                dtype=np.float32,
            )

            yield AudioFrame(
                samples=samples,
                sample_rate=chunk.sample_rate,
                channels=chunk.sample_channels,
            )

    def _open(self) -> None:
        if self._model is not None:
            raise PiperTTSError("Model is already loaded")

        with disable_progress_bars():
            self._logger.debug(
                "Loading model: model_path='{}'",
                self._model_path,
            )

            model_path = hf_hub_download(
                repo_id=self._repository_id,
                filename=self._model_path,
            )
            config_path = hf_hub_download(
                repo_id=self._repository_id,
                filename=self._model_path + ".json",
            )

            assert isinstance(model_path, str)
            assert isinstance(config_path, str)

            self._model = PiperVoice.load(
                model_path=model_path,
                config_path=config_path,
            )

        self._logger.debug("Model READY")

    def _close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._state = None
        self._logger.debug("Model CLOSED")

    def _require_model(self) -> PiperVoice:
        if self._model is None:
            raise PiperTTSError("Model is not loaded")

        return self._model
