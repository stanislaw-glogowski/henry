import numpy as np
import pyaudio
from loguru import logger

from ..domain import AudioFrame
from ..ports import InputStream, InputStreamError, StreamConfig


class PyAudioInputError(InputStreamError): ...


class PyAudioInput(InputStream):
    def __init__(
        self,
        driver: pyaudio.PyAudio,
        config: StreamConfig,
    ) -> None:
        self._driver = driver
        self._config = config
        self._stream: pyaudio.Stream | None = None
        self._logger = logger.bind(component="PyAudioInput")

    def read(self) -> AudioFrame:
        stream = self._require_stream()

        buffer = stream.read(
            self._config.frames_per_buffer,
            exception_on_overflow=False,
        )

        samples: np.ndarray = np.frombuffer(
            buffer,
            dtype=np.float32,
        )

        return self._config.build_frame(samples)

    def open(self) -> None:
        if self._stream is not None:
            raise PyAudioInputError("Stream is already open")

        self._stream = self._driver.open(
            rate=self._config.sample_rate,
            channels=self._config.channels,
            frames_per_buffer=self._config.frames_per_buffer,
            format=pyaudio.paFloat32,
            input=True,
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
            raise PyAudioInputError("Stream is not open")

        return self._stream
