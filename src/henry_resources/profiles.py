from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from pydantic import ConfigDict, Field

from henry_conversation.config import ConversationProfile
from henry_speech.config import SpeechProfile


class Profile(SpeechProfile):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    id: str = Field(exclude=True)
    path: Path = Field(exclude=True)
    name: str = Field(min_length=1)
    conversation: ConversationProfile

    @staticmethod
    def load_from_directory(path: Path) -> Profile:
        profile_path = path / "profile.yml"
        data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        conversation = data.get("conversation", {})
        profile = Profile.model_validate(
            {
                **data,
                "conversation": {
                    **conversation,
                    "prompts": {
                        "system": Profile._read_prompt(path, "system.md"),
                        "opening": Profile._read_prompt(path, "opening.md"),
                        "summary": Profile._read_prompt(path, "summary.md"),
                    },
                },
                "id": path.name,
                "path": path,
            }
        )
        return profile

    @staticmethod
    def _read_prompt(profile_path: Path, name: str) -> str:
        path = profile_path / "prompts" / name
        if not path.is_file():
            raise FileNotFoundError(f"Profile prompt file does not exist: {path}")
        return path.read_text(encoding="utf-8")


class ProfileCatalog(ABC):
    _PROFILES_PATH = "profiles"

    @abstractmethod
    def load_profile(self, name: str) -> Profile:
        raise NotImplementedError

    @abstractmethod
    def list_profiles(self) -> list[Profile]:
        raise NotImplementedError
