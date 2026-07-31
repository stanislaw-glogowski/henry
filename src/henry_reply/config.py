from pydantic import Field

from henry_common.validation import ConfigModel


class LLMProfile(ConfigModel):
    model: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)


class ReplyProfile(ConfigModel):
    reply: LLMProfile
