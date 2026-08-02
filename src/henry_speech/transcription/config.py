from typing import Annotated, Literal

from pydantic import Field

from henry_common.validation import ConfigModel

type STTSettings = Annotated[
    MLXParakeetTDTSettings | MLXQwen3ASRSettings | MLXWhisperSettings,
    Field(discriminator="adapter"),
]


# mlx


class MLXBaseSettings(ConfigModel):
    model_id: str = Field(min_length=1)


class MLXBaseProfile(ConfigModel):
    model_id: str | None = Field(min_length=1, default=None)


class MLXParakeetTDTSettings(MLXBaseSettings):
    adapter: Literal["mlx:parakeet-tdt"] = "mlx:parakeet-tdt"
    model_id: str = Field(
        min_length=1,
        default="mlx-community/parakeet-tdt-0.6b-v3",
    )


class MLXParakeetTDTProfile(MLXBaseProfile):
    pass


class MLXQwen3ASRSettings(MLXBaseSettings):
    adapter: Literal["mlx:qwen3-asr"] = "mlx:qwen3-asr"
    model_id: str = Field(
        min_length=1,
        default="mlx-community/Qwen3-ASR-0.6B-8bit",
    )


class MLXQwen3ASRProfile(MLXBaseProfile):
    pass


class MLXWhisperSettings(MLXBaseSettings):
    adapter: Literal["mlx:whisper"] = "mlx:whisper"
    model_id: str = Field(
        min_length=1,
        default="mlx-community/whisper-large-v3-turbo-asr-fp16",
    )


class MLXWhisperProfile(MLXBaseProfile):
    language: str | None = Field(default=None, min_length=1)


class STTProfile(ConfigModel):
    stt: dict[str, object] = Field(default_factory=dict)

    @property
    def stt_mlx_parakeet_tdt(self) -> MLXParakeetTDTProfile:
        return MLXParakeetTDTProfile.model_validate(self.stt)

    @property
    def stt_mlx_qwen3_asr(self) -> MLXQwen3ASRProfile:
        return MLXQwen3ASRProfile.model_validate(self.stt)

    @property
    def stt_mlx_whisper(self) -> MLXWhisperProfile:
        return MLXWhisperProfile.model_validate(self.stt)


def default_stt_settings() -> STTSettings:
    return MLXParakeetTDTSettings()
