import os
from pathlib import Path

from platformdirs import user_data_dir

from .models import ModelCatalog
from .profiles import Profile, ProfileCatalog
from .settings import Settings, SettingsStore


class LocalStore(ModelCatalog, ProfileCatalog, SettingsStore):
    _HOME_ENV_VAR = "HENRY_HOME"
    _LOCAL_DIR = ".henry"
    _USER_DIR = "Henry"

    def __init__(self, root_path: Path | None = None) -> None:
        self._root_path = (
            root_path if root_path is not None else self._locate_root_path()
        )

    def ensure_model_path(self, *paths: Path | str) -> Path:
        path = (self._root_path / self._MODELS_DIR).joinpath(*paths)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    def load_profile(self, name: str) -> Profile:
        path = self._profiles_path / (name + ".yml")
        if not path.is_file():
            raise FileNotFoundError(f"Profile not found: {path}")
        return Profile.load_from_file(path)

    def list_profiles(self) -> list[Profile]:
        if not self._profiles_path.is_dir():
            raise FileNotFoundError(f"Profiles not found: {self._profiles_path}")

        return [
            Profile.load_from_file(path) for path in self._profiles_path.glob("*.yml")
        ]

    def load_settings(self) -> Settings:
        path = self._root_path / self._SETTINGS_FILE
        if not path.is_file():
            raise FileNotFoundError(f"Settings not found: {path}")
        return Settings.load_from_file(path)

    @property
    def _profiles_path(self) -> Path:
        return self._root_path / self._PROFILES_PATH

    def _locate_root_path(self) -> Path:
        if value := os.getenv(self._HOME_ENV_VAR):
            return Path(value).expanduser()

        start = Path.cwd()
        for directory in (start, *start.parents):
            local = directory / self._LOCAL_DIR
            if local.is_dir():
                return local

        return Path(user_data_dir(self._USER_DIR))
