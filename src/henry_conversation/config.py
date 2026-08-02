from typing import Annotated, Literal

from pydantic import Field, model_validator

from henry_common.validation import ConfigModel


class LanguageModelProfile(ConfigModel):
    langchain: str | None = Field(default=None, min_length=1)
    mlx: str | None = Field(default=None, min_length=1)
    max_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0)
    thinking: bool = False

    @model_validator(mode="after")
    def validate_model(self) -> LanguageModelProfile:
        if self.langchain is None and self.mlx is None:
            raise ValueError("At least one language model adapter must be configured")
        return self

    def model_for(self, adapter: Literal["langchain", "mlx"]) -> str:
        if model := getattr(self, adapter):
            return model
        raise ValueError(f"No model is configured for the {adapter!r} adapter")


class LanguageModelsProfile(ConfigModel):
    fast: LanguageModelProfile
    detailed: LanguageModelProfile
    classifier: LanguageModelProfile | None = None


class ConversationPrompts(ConfigModel):
    system: str = Field(min_length=1)
    opening: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class ConversationReactions(ConfigModel):
    wake: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    wait: tuple[Annotated[str, Field(min_length=1)], ...] = ()


class ConversationProfile(ConfigModel):
    models: LanguageModelsProfile
    recent_messages: int = Field(default=8, ge=2)
    prompts: ConversationPrompts
    reactions: ConversationReactions = ConversationReactions()


class ConversationSettings(ConfigModel):
    adapter: Literal["langchain", "mlx"] = "langchain"
    acknowledgement_delay: float = Field(default=0.5, ge=0.0)
    classify_ambiguous: bool = False
