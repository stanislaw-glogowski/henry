from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from henry_reply.config import ReplyProfile
from henry_speech.config import SpeechProfile


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


class Profile(ReplyProfile, SpeechProfile):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    id: str = Field(exclude=True)
    path: Path = Field(exclude=True)
    name: str = Field(min_length=1)
    language: str = Field(min_length=1)

    @staticmethod
    def load_from_file(path: Path) -> Profile:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        profile = Profile.model_validate(
            {
                **data,
                "id": path.stem,
                "path": path,
            }
        )
        return profile


class ProfileCatalog(ABC):
    _PROFILES_PATH = "profiles"

    @abstractmethod
    def load_profile(self, name: str) -> Profile:
        raise NotImplementedError

    @abstractmethod
    def list_profiles(self) -> list[Profile]:
        raise NotImplementedError
