from typing import Literal

from pydantic import Field

from henry_common.validation import ConfigModel


class TTSProfile(ConfigModel):
    model: str = Field(min_length=1)


class TTSSettings(ConfigModel):
    adapter: Literal["piper", "mlx:chatterbox"] = "piper"
