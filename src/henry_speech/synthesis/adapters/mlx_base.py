from abc import ABC
from typing import ClassVar

from huggingface_hub.utils import disable_progress_bars

from ..config import MLXBaseProfile, MLXBaseSettings
from ..ports import TTSModel


class MLXBaseModel[TModel, TProfile: MLXBaseProfile, TSettings: MLXBaseSettings](
    TTSModel, ABC
):
    _MODEL_LABEL: ClassVar[str]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        if "_MODEL_LABEL" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define _MODEL_LABEL")

        if not isinstance(cls._MODEL_LABEL, str):
            raise TypeError("_MODEL_LABEL must be a string")

    def __init__(self, profile: TProfile, settings: TSettings) -> None:
        super().__init__()
        self._profile = profile
        self._settings = settings
        self._model: TModel | None = None

    def open(self) -> None:
        if self._model is not None:
            raise RuntimeError(f"{self._MODEL_LABEL} model is already loaded")

        with disable_progress_bars():
            from mlx_audio.tts.utils import load

            model_id = self._profile.model_id or self._settings.model_id

            self._logger.debug("Loading model: model_id='{}'", model_id)
            self._model = load(model_id)

        self._logger.debug("Model READY")

    def close(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._logger.debug("Model CLOSED")

    def _require_model(self) -> TModel:
        if self._model is None:
            raise RuntimeError(f"{self._MODEL_LABEL} model is not loaded")

        return self._model
