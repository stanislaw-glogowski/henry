from pathlib import Path

from .locator import locate_root

MODELS_DIR = "models"


def ensure_model_path(*path_parts: str) -> Path:
    path = locate_root() / MODELS_DIR
    path = path.joinpath(*path_parts)

    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    return path
