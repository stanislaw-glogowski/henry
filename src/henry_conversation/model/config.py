from typing import Annotated, Literal

from pydantic import Field

from henry_common.validation import ConfigModel


class LangChainSettings(ConfigModel):
    adapter: Literal["langchain"] = "langchain"
    base_url: str = Field(default="http://localhost:11434", min_length=1)


class MLXSettings(ConfigModel):
    adapter: Literal["mlx"] = "mlx"


type LanguageModelSettings = Annotated[
    LangChainSettings | MLXSettings,
    Field(discriminator="adapter"),
]


class BaseModelProfile(ConfigModel):
    model_id: str = Field(min_length=1)
    max_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    thinking: bool = False


class LangChainModelProfile(BaseModelProfile):
    pass


class MLXModelProfile(BaseModelProfile):
    top_k: int = Field(default=20, ge=0)


class LangChainModelsProfile(ConfigModel):
    fast: LangChainModelProfile
    detailed: LangChainModelProfile
    classifier: LangChainModelProfile | None = None


class MLXModelsProfile(ConfigModel):
    fast: MLXModelProfile
    detailed: MLXModelProfile
    classifier: MLXModelProfile | None = None


class LanguageModelProfile(ConfigModel):
    models: dict[str, object]

    @property
    def models_langchain(self) -> LangChainModelsProfile:
        return LangChainModelsProfile.model_validate(self.models)

    @property
    def models_mlx(self) -> MLXModelsProfile:
        return MLXModelsProfile.model_validate(self.models)


def default_language_model_settings() -> LanguageModelSettings:
    return LangChainSettings()
