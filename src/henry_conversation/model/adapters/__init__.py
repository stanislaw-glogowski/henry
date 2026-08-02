from typing import TYPE_CHECKING

from ..config import LangChainSettings, MLXSettings

if TYPE_CHECKING:
    from ..config import LanguageModelProfile, LanguageModelSettings
    from ..ports import LanguageModel


def get_language_model(
    profile: LanguageModelProfile,
    settings: LanguageModelSettings,
    *,
    require_classifier: bool = False,
) -> LanguageModel:
    match settings:
        case LangChainSettings():
            from .langchain import LangChainLanguageModel

            models = profile.models_langchain
            if require_classifier and models.classifier is None:
                raise ValueError(
                    "Ambiguous-turn classification requires a classifier model"
                )
            return LangChainLanguageModel(models, settings)
        case MLXSettings():
            from .mlx import MLXLanguageModel

            models = profile.models_mlx
            if require_classifier and models.classifier is None:
                raise ValueError(
                    "Ambiguous-turn classification requires a classifier model"
                )
            return MLXLanguageModel(models)
        case _:
            raise ValueError(f"Unsupported language model settings: {settings!r}")
