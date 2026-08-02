from typing import Annotated, Literal

from pydantic import Field

from henry_common.validation import ConfigModel

type TTSSettings = Annotated[
    PiperSettings | MLXChatterboxSettings,
    Field(discriminator="adapter"),
]


# piper


class PiperSettings(ConfigModel):
    adapter: Literal["piper"] = "piper"
    repo_id: str = Field(min_length=1, default="rhasspy/piper-voices")
    normalize_audio: bool = True
    volume: float = Field(default=1.0, gt=0.0)


class PiperProfile(ConfigModel):
    model_path: str = Field(
        min_length=1,
        default="en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    )
    repo_id: str | None = Field(min_length=1, default=None)
    speaker_id: int | None = Field(default=None, ge=0)
    length_scale: float | None = Field(default=None, gt=0.0)
    noise_scale: float | None = Field(default=None, ge=0.0)
    noise_w_scale: float | None = Field(default=None, ge=0.0)


# mlx


class MLXBaseSettings(ConfigModel):
    model_id: str = Field(min_length=1)


class MLXBaseProfile(ConfigModel):
    model_id: str | None = Field(min_length=1, default=None)


class MLXChatterboxSettings(MLXBaseSettings):
    adapter: Literal["mlx:chatterbox"] = "mlx:chatterbox"
    model_id: str = Field(min_length=1, default="mlx-community/chatterbox-fp16")
    lang_code: str = Field(min_length=1, default="en")


class MLXChatterboxProfile(MLXBaseProfile):
    lang_code: str | None = Field(min_length=1, default=None)


# configs


class TTSProfile(ConfigModel):
    tts: dict[str, object] = Field(default_factory=dict)

    @property
    def tts_piper(self) -> PiperProfile:
        return PiperProfile.model_validate(self.tts)

    @property
    def tts_mlx_chatterbox(self) -> MLXChatterboxProfile:
        return MLXChatterboxProfile.model_validate(self.tts)


def default_tts_settings() -> TTSSettings:
    return PiperSettings()
