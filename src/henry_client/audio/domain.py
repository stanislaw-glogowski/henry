from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type AudioSamples = NDArray[np.float32]


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

    def build_frame(self, samples: AudioSamples) -> AudioFrame:
        return AudioFrame(
            channels=self.channels,
            sample_rate=self.sample_rate,
            samples=samples,
        )


@dataclass(frozen=True, slots=True)
class AudioFrame(AudioFormat):
    samples: AudioSamples

    def build_chunk(self, is_speech: bool, vad_score: float) -> AudioChunk:
        return AudioChunk(
            channels=self.channels,
            sample_rate=self.sample_rate,
            samples=self.samples,
            is_speech=is_speech,
            vad_score=vad_score,
        )


@dataclass(frozen=True, slots=True)
class AudioChunk(AudioFrame):
    is_speech: bool
    vad_score: float


class AudioBuffer:
    def __init__(self) -> None:
        self._list: list[AudioSamples] = list()
        self._format: AudioFormat | None = None

    def __len__(self) -> int:
        return len(self._list)

    def append(self, frame: AudioFrame) -> None:
        if self._format is None:
            self._format = AudioFormat(
                sample_rate=frame.sample_rate,
                channels=frame.channels,
            )
        else:
            self._format.verify(frame)
        self._list.append(frame.samples)

    def build(self) -> AudioFrame | None:
        if self._format is None:
            return None

        return self._format.build_frame(np.concatenate(self._list))

    def clear(self) -> None:
        self._list.clear()
