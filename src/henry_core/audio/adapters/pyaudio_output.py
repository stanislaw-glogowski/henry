import numpy as np
import pyaudio
from loguru import logger

from ..domain import AudioFrame
from ..ports import OutputStream, OutputStreamError, StreamConfig


class PyAudioOutputError(OutputStreamError): ...


class PyAudioOutput(OutputStream):
    def __init__(
        self,
        driver: pyaudio.PyAudio,
        config: StreamConfig,
    ) -> None:
        self._driver = driver
        self._config = config
        self._stream: pyaudio.Stream | None = None
        self._logger = logger.bind(component="PyAudioOutput")

    def write(self, frame: AudioFrame) -> None:
        stream = self._require_stream()
        self._config.verify(frame)

        buffer = np.ascontiguousarray(
            frame.samples,
            dtype=np.float32,
        ).tobytes()

        stream.write(buffer)

    def open(self) -> None:
        if self._stream is not None:
            raise PyAudioOutputError("Stream is already open")

        self._stream = self._driver.open(
            rate=self._config.sample_rate,
            channels=self._config.channels,
            frames_per_buffer=self._config.frames_per_buffer,
            format=pyaudio.paFloat32,
            output=True,
        )

        self._logger.trace(
            "Stream OPENED: sample_rate={}, channels={}, frames_per_buffer={}",
            self._config.sample_rate,
            self._config.channels,
            self._config.frames_per_buffer,
        )

    def close(self) -> None:
        stream = self._stream

        if stream is None:
            return

        if stream.is_active():
            stream.stop_stream()

        stream.close()

        self._stream = None
        self._logger.trace("Stream CLOSED")

    def _require_stream(self) -> pyaudio.Stream:
        if self._stream is None:
            raise PyAudioOutputError("Stream is not open")

        return self._stream
