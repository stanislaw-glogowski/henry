from typing import Literal

from pydantic import Field

from henry_common.validation import ConfigModel


class VADSettings(ConfigModel):
    adapter: Literal["mlx:silero_vad", "openwakeword"] = "mlx:silero_vad"
    threshold: float = 0.5


class WakeWordProfile(ConfigModel):
    model: str = Field(min_length=1)
    threshold: float = 0.75


class WakeWordSettings(ConfigModel):
    adapter: Literal["openwakeword"] = "openwakeword"
