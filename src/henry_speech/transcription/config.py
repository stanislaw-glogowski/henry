from typing import Literal

from henry_common.validation import ConfigModel


class STTProfile(ConfigModel):
    model: str | None = None


class STTSettings(ConfigModel):
    adapter: Literal["mlx:parakeet-tdt"] = "mlx:parakeet-tdt"
