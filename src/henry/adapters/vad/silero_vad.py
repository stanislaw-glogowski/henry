from types import TracebackType
from typing import Self

from huggingface_hub.utils import disable_progress_bars
from loguru import logger
from mlx_audio.vad import load
from mlx_audio.vad.models.silero_vad import Model as VadModel
from mlx_audio.vad.models.silero_vad import SileroVADState

from ...domain import AudioFrame, SpeechChunk
from ...ports import VoiceActivityDetector

MODEL_ID = "mlx-community/silero-vad"


class SileroVad(VoiceActivityDetector):
    def __init__(
        self,
        threshold: float,
    ) -> None:
        self._threshold = threshold
        self._model: VadModel | None = None
        self._state: SileroVADState | None = None
        self._logger = logger.bind(component="SileroVad")

    def __enter__(self) -> Self:
        with disable_progress_bars():
            self._logger.trace("Loading model: model_id='{}'", MODEL_ID)
            model = load(MODEL_ID)
            assert isinstance(model, VadModel)
            self._model = model

        self._logger.trace("Model READY")

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def analyze(self, frame: AudioFrame) -> SpeechChunk:
        model = self._require_model()

        probability, self._state = model.feed(
            frame.samples,
            self._state,
            sample_rate=frame.format.sample_rate,
        )

        score = float(probability.item())

        return SpeechChunk(
            audio=frame,
            speech_score=score,
            is_speech=score >= self._threshold,
        )

    def _require_model(self) -> VadModel:
        if self._model is None:
            raise RuntimeError("Silero VAD is not loaded")

        return self._model
