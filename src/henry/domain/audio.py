from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    channels: int = 1


@dataclass(frozen=True, slots=True)
class AudioFrame:
    samples: NDArray[np.float32]
    format: AudioFormat


@dataclass(slots=True)
class AudioBuffer:
    chunks: list[NDArray[np.float32]] = field(default_factory=list)
    format: AudioFormat | None = None

    def append(self, frame: AudioFrame) -> None:
        if self.format is None:
            self.format = frame.format
        elif frame.format != self.format:
            raise RuntimeError("Invalid frame format")
        self.chunks.append(frame.samples)

    def build(self) -> AudioFrame | None:
        if self.format is None:
            return None
        return AudioFrame(
            samples=np.concatenate(self.chunks),
            format=self.format,
        )

    def clear(self) -> None:
        self.chunks.clear()
