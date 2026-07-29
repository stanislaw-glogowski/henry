from dataclasses import dataclass

import numpy as np

type AudioSamples = np.ndarray


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    channels: int

    def verify(self, other_format: AudioFormat) -> None:
        """Raise when channels or sample rate differ from this format."""
        if other_format.channels != self.channels:
            raise RuntimeError(
                f"Incompatible channel count: expected {self.channels}, "
                f"got {other_format.channels}"
            )
        if other_format.sample_rate != self.sample_rate:
            raise RuntimeError(
                f"Incompatible sample rate: expected {self.sample_rate}, "
                f"got {other_format.sample_rate}"
            )

    def build_frame(self, samples: AudioSamples | bytes) -> AudioFrame:
        match samples:
            case bytes():
                samples = np.frombuffer(samples, dtype=np.float32)

        return AudioFrame(
            format=self,
            samples=samples,
        )


@dataclass(frozen=True, slots=True)
class AudioFrame:
    format: AudioFormat
    samples: AudioSamples

    @property
    def samples_count(self) -> int:
        return len(self.samples)

    def to_bytes(self) -> bytes:
        return np.ascontiguousarray(
            self.samples,
            dtype=np.float32,
        ).tobytes()


class AudioBuffer:
    """Accumulate frames of one format until built or cleared."""

    def __init__(self) -> None:
        self._list: list[AudioSamples] = list()
        self._format: AudioFormat | None = None

    def __len__(self) -> int:
        return len(self._list)

    def append(self, frame: AudioFrame) -> None:
        if self._format is None:
            self._format = frame.format
        else:
            self._format.verify(frame.format)
        self._list.append(frame.samples)

    def build(self) -> AudioFrame | None:
        if self._format is None:
            return None

        return self._format.build_frame(np.concatenate(self._list))

    def clear(self) -> None:
        self._list.clear()
        self._format = None
