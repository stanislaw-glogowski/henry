from collections.abc import Iterator

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from piper import PiperVoice, SynthesisConfig

from ...audio import AudioFormat, AudioFrame
from ..config import PiperProfile, PiperSettings
from ..ports import TTSModel


class PiperModel(TTSModel):
    def __init__(
        self,
        profile: PiperProfile,
        settings: PiperSettings,
    ) -> None:
        super().__init__()
        self._profile = profile
        self._settings = settings
        self._model: PiperVoice | None = None

    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        if self._model is None:
            raise RuntimeError("Piper voice model is not loaded")

        chunks = self._model.synthesize(
            text=text,
            include_alignments=False,
            syn_config=SynthesisConfig(
                speaker_id=self._profile.speaker_id,
                length_scale=self._profile.length_scale,
                noise_scale=self._profile.noise_scale,
                noise_w_scale=self._profile.noise_w_scale,
                normalize_audio=self._settings.normalize_audio,
                volume=self._settings.volume,
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
            repo_id = self._profile.repo_id or self._settings.repo_id
            filename = self._profile.voice_path

            self._logger.debug(
                "Loading model: model_path='{}/{}'",
                repo_id,
                filename,
            )

            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
            config_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename + ".json",
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
