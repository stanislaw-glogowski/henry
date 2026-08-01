from typing import Literal

from pydantic import Field

from henry_common.validation import ConfigModel


class STTProfile(ConfigModel):
    model: str | None = None
    language: str | None = Field(default=None, min_length=1)


class STTSettings(ConfigModel):
    adapter: Literal[
        "mlx:parakeet-tdt",
        "mlx:qwen3-asr",
        "mlx:whisper",
    ] = "mlx:parakeet-tdt"
