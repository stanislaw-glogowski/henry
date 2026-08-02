from pydantic import Field

from henry_common.validation import ConfigModel

from .model.config import LanguageModelSettings, default_language_model_settings


class ConversationSettings(ConfigModel):
    model: LanguageModelSettings = default_language_model_settings()
    acknowledgement_delay: float = Field(default=0.5, ge=0.0)
    classify_ambiguous: bool = False
