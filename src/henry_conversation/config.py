from pydantic import Field

from henry_common.validation import ConfigModel


class ConversationPrompts(ConfigModel):
    system: str = Field(min_length=1)
    opening: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class ConversationProfile(ConfigModel):
    model: str = Field(min_length=1)
    recent_messages: int = Field(default=8, ge=2)
    prompts: ConversationPrompts
