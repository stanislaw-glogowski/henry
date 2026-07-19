from huggingface_hub.utils import disable_progress_bars
from lifecycle import ManagedResource
from loguru import logger
from mlx_audio.vad import load
from mlx_audio.vad.models.silero_vad import Model as VadModel
from mlx_audio.vad.models.silero_vad import SileroVADState

from ...audio import AudioFrame
from ..domain import VAD

MODEL_ID = "mlx-community/silero-vad"


class SileroVAD(VAD, ManagedResource):
    def __init__(self) -> None:
        self._model: VadModel | None = None
        self._state: SileroVADState | None = None
        self._logger = logger.bind(component="SileroVad")

    def predict(self, frame: AudioFrame) -> float:
        model = self._require_model()

        probability, self._state = model.feed(
            frame.samples,
            self._state,
            sample_rate=frame.sample_rate,
        )

        return float(probability.item())

    def open(self) -> None:
        if self._model is not None:
            return

        with disable_progress_bars():
            self._logger.trace("Loading model: model_id='{}'", MODEL_ID)
            model = load(MODEL_ID)
            assert isinstance(model, VadModel)
            self._model = model

        self._logger.trace("Model READY")

    def close(self) -> None:
        pass

    def _require_model(self) -> VadModel:
        if self._model is None:
            raise RuntimeError("Silero VAD is not loaded")

        return self._model
