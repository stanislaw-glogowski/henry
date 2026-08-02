from typing import Annotated

from pydantic import Field

from henry_common.validation import ConfigModel

from ..model.config import LanguageModelProfile


class ConversationPrompts(ConfigModel):
    system: str = Field(min_length=1)
    opening: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class ConversationReactions(ConfigModel):
    wake: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    wait: tuple[Annotated[str, Field(min_length=1)], ...] = ()


class ConversationProfile(LanguageModelProfile):
    recent_messages: int = Field(default=8, ge=2)
    prompts: ConversationPrompts
    reactions: ConversationReactions = ConversationReactions()
