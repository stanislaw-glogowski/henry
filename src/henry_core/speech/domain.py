from abc import ABC, abstractmethod

from ..audio import AudioFrame


class VAD(ABC):
    @abstractmethod
    def predict(self, frame: AudioFrame) -> float:
        raise NotImplementedError
