from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

type AudioSamples = np.ndarray


@dataclass(frozen=True, slots=True)
class AudioDevice:
    """Read-only device identity reported by an audio adapter."""

    name: str
    identifier: str | None = None


@dataclass(frozen=True, slots=True)
class AudioDevices:
    """Input and output devices selected for the open audio session."""

    input: AudioDevice
    output: AudioDevice


class AudioPlaybackOutcome(Enum):
    """Device acknowledgement for one submitted audio frame."""

    PLAYED = auto()
    INTERRUPTED = auto()


@dataclass(frozen=True, slots=True)
class AudioFormat:
    sample_rate: int
    channels: int

    def verify(self, other_format: AudioFormat) -> None:
        """Raise when channels or sample rate differ from this format."""
        if other_format.channels != self.channels:
            raise RuntimeError(
                f"Audio channel mismatch: expected {self.channels}, "
                f"got {other_format.channels}"
            )
        if other_format.sample_rate != self.sample_rate:
            raise RuntimeError(
                f"Audio sample rate mismatch: expected {self.sample_rate} Hz, "
                f"got {other_format.sample_rate} Hz"
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
