from collections.abc import Iterator

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from piper import PiperVoice, SynthesisConfig

from ...audio import AudioFormat, AudioFrame
from ..config import TTSProfile
from ..ports import TTSModel


class PiperTTSModel(TTSModel):
    _REPOSITORY_ID: str = "rhasspy/piper-voices"
    _LENGTH_SCALE: float = 1.05
    _NOISE_SCALE: float | None = None
    _NOISE_W_SCALE: float | None = None

    def __init__(
        self,
        profile: TTSProfile,
    ) -> None:
        super().__init__()
        self._profile = profile
        self._model: PiperVoice | None = None

    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        if self._model is None:
            raise RuntimeError("Piper voice model is not loaded")

        chunks = self._model.synthesize(
            text=text,
            include_alignments=False,
            syn_config=SynthesisConfig(
                length_scale=self._LENGTH_SCALE,
                noise_scale=self._NOISE_SCALE,
                noise_w_scale=self._NOISE_W_SCALE,
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
                format=AudioFormat(
                    sample_rate=chunk.sample_rate,
                    channels=chunk.sample_channels,
                ),
            )

    def open(self) -> None:
        if self._model is not None:
            raise RuntimeError("Piper voice model is already loaded")

        with disable_progress_bars():
            self._logger.debug(
                "Loading model: model_path='{}'",
                self._profile.model,
            )

            model_path = hf_hub_download(
                repo_id=self._REPOSITORY_ID,
                filename=self._profile.model,
            )
            config_path = hf_hub_download(
                repo_id=self._REPOSITORY_ID,
                filename=self._profile.model + ".json",
            )

            assert isinstance(model_path, str)
            assert isinstance(config_path, str)

            self._model = PiperVoice.load(
                model_path=model_path,
                config_path=config_path,
            )

        self._logger.debug("Model READY")

    def close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._logger.debug("Model CLOSED")
