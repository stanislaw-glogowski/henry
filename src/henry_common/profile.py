from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .storage import PathLocator


class ProfileModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class WakewordProfile(ProfileModel):
    model: str = Field(min_length=1)

    @field_validator("model")
    @classmethod
    def validate_model_extension(cls, value: str) -> str:
        if not value.endswith(".onnx"):
            raise ValueError("wakeword.model must be an ONNX file")
        return value


class VoiceProfile(ProfileModel):
    model: str = Field(min_length=1)


class ReplyProfile(ProfileModel):
    system_prompt: str = Field(min_length=1)
    model: str = Field(min_length=1)


class Profile(ProfileModel):
    path: Path = Field(exclude=True)
    name: str = Field(min_length=1)
    language: str = Field(min_length=1)
    wakeword: WakewordProfile
    voice: VoiceProfile
    reply: ReplyProfile

    @staticmethod
    def load(path: Path) -> Profile:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        profile = Profile.model_validate(
            {
                **data,
                "path": path,
            }
        )
        return profile


def load_profile(locator: PathLocator, name: str) -> Profile:
    path = locator.ensure_profiles_path() / (name + ".yml")
    return Profile.load(path)


def load_profiles(locator: PathLocator) -> dict[str, Profile]:
    profiles_path = locator.ensure_profiles_path()

    return {path.stem: Profile.load(path) for path in profiles_path.glob("*.yml")}
