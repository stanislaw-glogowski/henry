from typing import Literal

from pydantic import Field, field_validator

from henry_common.validation import ConfigModel


class VADSettings(ConfigModel):
    adapter: Literal["mlx:silero_vad", "openwakeword"] = "mlx:silero_vad"
    threshold: float = 0.5


class WakeWordProfile(ConfigModel):
    model: str = Field(min_length=1)
    threshold: float = 0.75

    @field_validator("model")
    @classmethod
    def validate_model_extension(cls, value: str) -> str:
        if not value.endswith(".onnx"):
            raise ValueError("wakeword.model must be an ONNX file")
        return value


class WakeWordSettings(ConfigModel):
    adapter: Literal["openwakeword"] = "openwakeword"
