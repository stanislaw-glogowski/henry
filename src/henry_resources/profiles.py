from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import ConfigDict, Field

from henry_conversation.profile import ConversationProfile
from henry_speech.config import SpeechProfile


class Profile(SpeechProfile):
    _PROMPTS_DIR: ClassVar[str] = "prompts"
    _REACTIONS_DIR: ClassVar[str] = "reactions"

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
        if not isinstance(data, dict):
            raise ValueError(f"Profile configuration must be a mapping: {profile_path}")

        conversation = data.get("conversation", {})
        if not isinstance(conversation, dict):
            raise ValueError(
                f"Profile conversation configuration must be a mapping: {profile_path}"
            )

        return Profile.model_validate(
            {
                **data,
                "conversation": {
                    **conversation,
                    "prompts": {
                        "system": Profile._read_prompt(path, "system.md"),
                        "opening": Profile._read_prompt(path, "opening.md"),
                        "summary": Profile._read_prompt(path, "summary.md"),
                    },
                    "reactions": {
                        "wait": Profile._read_reaction(path, "wait.txt"),
                        "wake": Profile._read_reaction(path, "wake.txt"),
                    },
                },
                "id": path.name,
                "path": path,
            }
        )

    @staticmethod
    def _read_prompt(profile_path: Path, name: str) -> str:
        path = profile_path / Profile._PROMPTS_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Profile prompt file does not exist: {path}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _read_reaction(profile_path: Path, name: str) -> tuple[str, ...]:
        path = profile_path / Profile._REACTIONS_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Profile reaction file does not exist: {path}")
        content = path.read_text(encoding="utf-8")

        return tuple(line.strip() for line in content.splitlines() if line.strip())


@dataclass(frozen=True, slots=True)
class ProfileEntry:
    id: str
    name: str
    profile: Profile | None = None
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.profile is not None


class ProfileCatalog(ABC):
    _PROFILES_PATH = "profiles"

    @abstractmethod
    def load_profile(self, name: str) -> Profile:
        raise NotImplementedError

    @abstractmethod
    def list_profiles(self) -> list[Profile]:
        raise NotImplementedError

    @abstractmethod
    def inspect_profiles(self) -> list[ProfileEntry]:
        raise NotImplementedError
