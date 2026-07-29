import os
from pathlib import Path

from platformdirs import user_data_dir


class PathLocator:
    _HOME_ENV_VAR = "HENRY_HOME"
    _LOCAL_DIR = ".henry"
    _USER_DIR = "Henry"
    _PROFILES_DIR = "profiles"
    _MODELS_DIR = "models"

    def __init__(self):
        self._root_path = self._locate_root_path()

    def ensure_model_path(self, *paths: Path | str) -> Path:
        model_path = (self._root_path / self._MODELS_DIR).joinpath(*paths)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return model_path

    def ensure_profiles_path(self, *paths: Path | str) -> Path:
        profiles_path = (self._root_path / self._MODELS_DIR).joinpath(*paths)
        if not profiles_path.is_dir():
            raise FileNotFoundError(f"Profiles not found: {profiles_path}")
        return profiles_path

    def _locate_root_path(self) -> Path:
        if value := os.getenv(self._HOME_ENV_VAR):
            return Path(value).expanduser()

        if local := self._locate_local_path(Path.cwd()):
            return local

        return Path(user_data_dir(self._USER_DIR))

    def _locate_local_path(self, start: Path) -> Path | None:
        for directory in (start, *start.parents):
            local = directory / self._LOCAL_DIR
            if local.is_dir():
                return local

        return None
