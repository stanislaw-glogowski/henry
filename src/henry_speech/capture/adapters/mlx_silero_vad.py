from typing import TYPE_CHECKING

from huggingface_hub.utils import disable_progress_bars

from ...audio import AudioFrame
from ..config import VADSettings
from ..domain import DetectionResult
from ..ports import VADModel

if TYPE_CHECKING:
    from mlx_audio.vad.models.silero_vad import Model
    from mlx_audio.vad.models.silero_vad import SileroVADState as ModelState


class SileroVADModel(VADModel):
    _MODEL_ID = "mlx-community/silero-vad"

    def __init__(
        self,
        settings: VADSettings | None = None,
    ) -> None:
        super().__init__("MLX")
        self._settings = settings if settings is not None else VADSettings()
        self._model: Model | None = None
        self._state: ModelState | None = None

    def detect(self, frame: AudioFrame) -> DetectionResult:
        if self._model is None:
            raise RuntimeError("MLX Silero VAD model is not loaded")

        probability, self._state = self._model.feed(
            frame.samples,
            self._state,
            sample_rate=frame.format.sample_rate,
        )

        score = float(probability.item())
        return DetectionResult(
            score=score,
            detected=score > self._settings.threshold,
        )

    def open(self) -> None:
        if self._model is not None:
            raise RuntimeError("MLX Silero VAD model is already loaded")

        with disable_progress_bars():
            from mlx_audio.vad import load
            from mlx_audio.vad.models.silero_vad import Model as LoadedModel

            self._logger.debug("Loading model: model_id='{}'", self._MODEL_ID)
            model = load(self._MODEL_ID)
            assert isinstance(model, LoadedModel)
            self._model = model

        self._logger.debug("Model READY")

    def close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._state = None
        self._logger.debug("Model CLOSED")
