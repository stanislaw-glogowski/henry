import os
from pathlib import Path

import yaml
from platformdirs import user_data_dir

from .models import ModelCatalog
from .profiles import Profile, ProfileCatalog, ProfileEntry
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
            raise FileNotFoundError(f"Model file does not exist: {path}")
        return path

    def load_profile(self, name: str) -> Profile:
        path = self._profiles_path / name
        if not path.is_dir():
            raise FileNotFoundError(f"Profile directory does not exist: {path}")
        return Profile.load_from_directory(path)

    def list_profiles(self) -> list[Profile]:
        if not self._profiles_path.is_dir():
            raise FileNotFoundError(
                f"Profiles directory does not exist: {self._profiles_path}"
            )

        return [
            Profile.load_from_directory(path)
            for path in sorted(self._profiles_path.iterdir())
            if path.is_dir()
        ]

    def inspect_profiles(self) -> list[ProfileEntry]:
        if not self._profiles_path.is_dir():
            return []

        entries: list[ProfileEntry] = []
        for path in sorted(self._profiles_path.iterdir()):
            if not path.is_dir():
                continue
            try:
                profile = Profile.load_from_directory(path)
            except Exception as error:
                entries.append(
                    ProfileEntry(
                        id=path.name,
                        name=self._profile_name(path),
                        error=str(error),
                    )
                )
            else:
                entries.append(
                    ProfileEntry(
                        id=profile.id,
                        name=profile.name,
                        profile=profile,
                    )
                )
        return entries

    def load_settings(self) -> Settings:
        path = self._root_path / self._SETTINGS_FILE
        if not path.is_file():
            raise FileNotFoundError(f"Settings file does not exist: {path}")
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

    @staticmethod
    def _profile_name(path: Path) -> str:
        profile_path = path / "profile.yml"
        if not profile_path.is_file():
            return path.name
        try:
            data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        except OSError, yaml.YAMLError:
            return path.name
        if isinstance(data, dict) and isinstance(data.get("name"), str):
            name = data["name"].strip()
            if name:
                return name
        return path.name
