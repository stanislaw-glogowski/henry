from typing import Literal

from pydantic import Field, field_validator

from henry_common.validation import ConfigModel

MLX_SILERO_VAD_MODEL_ID = "mlx-community/silero-vad"


class VADSettings(ConfigModel):
    adapter: Literal[
        "mlx:silero_vad",
        "openwakeword",
    ] = "mlx:silero_vad"
    threshold: float = 0.5


class WakeWordProfile(ConfigModel):
    label: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    threshold: float = 0.75

    @field_validator("model_path")
    @classmethod
    def validate_model_extension(cls, value: str) -> str:
        if not value.endswith(".onnx"):
            raise ValueError(f"wake-word model must be an ONNX file; got {value!r}")
        return value


class WakeWordSettings(ConfigModel):
    adapter: Literal["openwakeword"] = "openwakeword"
