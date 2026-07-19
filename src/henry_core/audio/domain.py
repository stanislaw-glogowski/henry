from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class AudioFormatError(Exception): ...


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    channels: int

    def verify(self, other_format: AudioFormat) -> None:
        if other_format.channels != self.channels:
            raise AudioFormatError(
                f"Incompatible channel count: expected {self.channels}, "
                f"got {other_format.channels}"
            )
        if other_format.sample_rate != self.sample_rate:
            raise AudioFormatError(
                f"Incompatible sample rate: expected {self.sample_rate}, "
                f"got {other_format.sample_rate}"
            )

    def build_frame(self, samples: NDArray[np.float32]) -> AudioFrame:
        return AudioFrame(
            channels=self.channels,
            sample_rate=self.sample_rate,
            samples=samples,
        )


@dataclass(frozen=True, slots=True)
class AudioFrame(AudioFormat):
    samples: NDArray[np.float32]

    def build_chunk(self, is_speech: bool) -> AudioChunk:
        return AudioChunk(
            channels=self.channels,
            sample_rate=self.sample_rate,
            samples=self.samples,
            is_speech=is_speech,
        )


@dataclass(frozen=True, slots=True)
class AudioChunk(AudioFrame):
    is_speech: bool
