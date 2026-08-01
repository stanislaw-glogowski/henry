import numpy as np

from ...domain import AudioFormat, AudioFrame
from ...ports import AudioInput
from ...resampler import AudioResampler
from .process import AVFAudioProcess


class AVFAudioInput(AudioInput):
    """Expose native capture as fixed-size 16 kHz mono frames."""

    _OUTPUT_FORMAT = AudioFormat(sample_rate=16_000, channels=1)
    _OUTPUT_FRAME_SIZE = 512

    def __init__(self, process: AVFAudioProcess) -> None:
        super().__init__()
        self._process = process
        self._resampler = AudioResampler(self._OUTPUT_FORMAT)
        self._samples = np.empty(0, dtype=np.float32)
        self._opened = False

    def read(self) -> AudioFrame:
        if not self._opened:
            raise RuntimeError("AVFAudio input stream is not open")

        while self._samples.size < self._OUTPUT_FRAME_SIZE:
            converted = self._resampler.process(self._process.read())
            if converted.samples.size:
                self._samples = np.concatenate((self._samples, converted.samples))

        samples = np.ascontiguousarray(
            self._samples[: self._OUTPUT_FRAME_SIZE],
            dtype=np.float32,
        )
        self._samples = self._samples[self._OUTPUT_FRAME_SIZE :].copy()
        return self._OUTPUT_FORMAT.build_frame(samples)

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("AVFAudio input stream is already open")

        self._resampler.reset()
        self._samples = np.empty(0, dtype=np.float32)
        self._opened = True
        self._logger.debug(
            "Stream OPENED: sample_rate={}, channels={}, frame_size={}",
            self._OUTPUT_FORMAT.sample_rate,
            self._OUTPUT_FORMAT.channels,
            self._OUTPUT_FRAME_SIZE,
        )

    def close(self) -> None:
        if not self._opened:
            return

        self._opened = False
        self._resampler.reset()
        self._samples = np.empty(0, dtype=np.float32)
        self._logger.debug("Stream CLOSED")
