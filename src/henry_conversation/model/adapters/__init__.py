from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config import ConversationProfile, ConversationSettings
    from ..ports import LanguageModel


def get_language_model(
    profile: ConversationProfile,
    settings: ConversationSettings,
) -> LanguageModel:
    if settings.adapter not in ("langchain", "mlx"):
        raise ValueError(f"Unsupported language model adapter: {settings.adapter!r}")

    profile.models.fast.model_for(settings.adapter)
    profile.models.detailed.model_for(settings.adapter)
    if settings.classify_ambiguous:
        if profile.models.classifier is None:
            raise ValueError(
                "Ambiguous-turn classification requires a classifier model"
            )
        profile.models.classifier.model_for(settings.adapter)

    match settings.adapter:
        case "langchain":
            from .langchain import LangChainLanguageModel

            return LangChainLanguageModel(profile.models)
        case "mlx":
            from .mlx import MLXLanguageModel

            return MLXLanguageModel(profile.models)
