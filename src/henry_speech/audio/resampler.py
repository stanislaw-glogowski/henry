import numpy as np
import soxr

from .domain import AudioFormat, AudioFrame


class AudioResampler:
    """Resample consecutive audio frames while preserving filter state."""

    def __init__(self, target_format: AudioFormat) -> None:
        if target_format.sample_rate <= 0:
            raise ValueError(
                f"Target sample rate must be positive; got {target_format.sample_rate}"
            )
        if target_format.channels <= 0:
            raise ValueError(
                f"Target channel count must be positive; got {target_format.channels}"
            )
        self._target_format = target_format
        self._source_format: AudioFormat | None = None
        self._stream: soxr.ResampleStream | None = None

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Resample the next frame without ending the continuous stream."""
        self._prepare(frame.format)
        samples = np.asarray(frame.samples, dtype=np.float32).reshape(-1)
        if samples.size % frame.format.channels:
            raise ValueError(
                "Audio samples are not aligned to the source channel count: "
                f"samples={samples.size}, channels={frame.format.channels}"
            )

        if self._stream is None:
            converted = samples
        else:
            source = samples.reshape(-1, frame.format.channels)
            converted = self._stream.resample_chunk(source, last=False)

        return self._target_format.build_frame(
            np.ascontiguousarray(converted, dtype=np.float32).reshape(-1)
        )

    def reset(self) -> None:
        """Discard the current source format and filter state."""
        self._source_format = None
        self._stream = None

    def _prepare(self, source_format: AudioFormat) -> None:
        if self._source_format is not None:
            self._source_format.verify(source_format)
            return
        if source_format.channels != self._target_format.channels:
            raise RuntimeError(
                "Audio resampling cannot change channel count: "
                f"source={source_format.channels}, "
                f"target={self._target_format.channels}"
            )

        self._source_format = source_format
        if source_format.sample_rate != self._target_format.sample_rate:
            self._stream = soxr.ResampleStream(
                source_format.sample_rate,
                self._target_format.sample_rate,
                source_format.channels,
                dtype="float32",
                quality="MQ",
            )
