from abc import ABC, abstractmethod
from pathlib import Path


class ModelCatalog(ABC):
    _MODELS_DIR = "models"

    @abstractmethod
    def ensure_model_path(self, *paths: Path | str) -> Path:
        raise NotImplementedError
