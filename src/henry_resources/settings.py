from abc import ABC, abstractmethod
from pathlib import Path

import yaml

from henry_common.validation import ConfigModel
from henry_conversation.config import ConversationSettings
from henry_speech.config import SpeechSettings


class Settings(ConfigModel):
    conversation: ConversationSettings = ConversationSettings()
    speech: SpeechSettings = SpeechSettings()

    @staticmethod
    def load_from_file(path: Path) -> Settings:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        settings = Settings.model_validate(data)
        return settings


class SettingsStore(ABC):
    _SETTINGS_FILE = "settings.yml"

    @abstractmethod
    def load_settings(self) -> Settings:
        raise NotImplementedError
