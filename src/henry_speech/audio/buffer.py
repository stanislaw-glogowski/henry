import numpy as np

from .domain import AudioFormat, AudioFrame, AudioSamples


class AudioBuffer:
    """Accumulate frames of one format until built or cleared."""

    def __init__(self) -> None:
        self._list: list[AudioSamples] = []
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
