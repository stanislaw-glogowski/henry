import os
from pathlib import Path

from platformdirs import user_data_dir

HOME_ENV_VAR = "HENRY_HOME"
LOCAL_DIR = ".henry"
USER_DIR = "Henry"


def locate_local(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        local = directory / LOCAL_DIR
        if local.is_dir():
            return local

    return None


def locate_root() -> Path:
    if value := os.getenv(HOME_ENV_VAR):
        return Path(value).expanduser()

    if local := locate_local(Path.cwd()):
        return local

    return Path(user_data_dir(USER_DIR))
