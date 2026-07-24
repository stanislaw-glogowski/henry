from dataclasses import dataclass
from enum import StrEnum

import pyaudio
from loguru import logger

from ..domain import AudioConfig, AudioFormat, AudioFrame
from ..ports import InputStream, OutputStream
from .pyaudio_session import PyAudioSession


class PyAudioStreamMode(StrEnum):
    DUPLEX = "duplex"
    INPUT = "input"
    OUTPUT = "output"

    @property
    def is_duplex(self) -> bool:
        return self is PyAudioStreamMode.DUPLEX

    @property
    def is_input(self) -> bool:
        return self.is_duplex or self is PyAudioStreamMode.INPUT

    @property
    def is_output(self) -> bool:
        return self.is_duplex or self is PyAudioStreamMode.OUTPUT


@dataclass(frozen=True, slots=True)
class PyAudioStreamConfig(AudioConfig):
    mode: PyAudioStreamMode = PyAudioStreamMode.DUPLEX


class PyAudioStreamError(RuntimeError): ...


class PyAudioStream(InputStream, OutputStream):
    _DUPLEX_SAMPLE_RATE = 16_000
    _INPUT_SAMPLE_RATE = 16_000
    _OUTPUT_SAMPLE_RATE = 22_050

    @staticmethod
    def duplex(session: PyAudioSession) -> PyAudioStream:
        return PyAudioStream(
            session,
            PyAudioStreamConfig(
                format=AudioFormat(PyAudioStream._DUPLEX_SAMPLE_RATE),
                mode=PyAudioStreamMode.DUPLEX,
            ),
        )

    @staticmethod
    def input(session: PyAudioSession) -> PyAudioStream:
        return PyAudioStream(
            session,
            PyAudioStreamConfig(
                format=AudioFormat(PyAudioStream._INPUT_SAMPLE_RATE),
                mode=PyAudioStreamMode.INPUT,
            ),
        )

    @staticmethod
    def output(session: PyAudioSession) -> PyAudioStream:
        return PyAudioStream(
            session,
            PyAudioStreamConfig(
                format=AudioFormat(PyAudioStream._OUTPUT_SAMPLE_RATE),
                mode=PyAudioStreamMode.OUTPUT,
            ),
        )

    def __init__(
        self,
        session: PyAudioSession,
        config: PyAudioStreamConfig,
    ) -> None:
        self._session = session
        self._config = config
        self._stream: pyaudio.Stream | None = None
        self._logger = logger.bind(
            component=f"PyAudioStream({self._config.mode.value})",
        )

    def read(self) -> AudioFrame:
        stream = self._require_stream(PyAudioStreamMode.INPUT)

        buffer = stream.read(
            self._config.frames_per_buffer,
            exception_on_overflow=False,
        )

        return self._config.format.build_frame(buffer)

    def write(self, frame: AudioFrame) -> None:
        stream = self._require_stream(PyAudioStreamMode.OUTPUT)
        self._config.format.verify(frame.format)
        stream.write(frame.to_bytes())

    def open(self) -> None:
        if self._stream is not None:
            raise PyAudioStreamError("Stream is already open")

        self._stream = self._session.driver.open(
            rate=self._config.format.sample_rate,
            channels=self._config.format.channels,
            frames_per_buffer=self._config.frames_per_buffer,
            input=self._config.mode.is_input,
            output=self._config.mode.is_output,
            format=pyaudio.paFloat32,
        )

        self._logger.debug(
            "Stream OPENED: sample_rate={}, channels={}, frames_per_buffer={}, mode={}",
            self._config.format.sample_rate,
            self._config.format.channels,
            self._config.frames_per_buffer,
            self._config.mode.value,
        )

    def close(self) -> None:
        stream = self._stream

        if stream is None:
            return

        if stream.is_active():
            stream.stop_stream()

        stream.close()

        self._stream = None
        self._logger.debug("Stream CLOSED")

    def _require_stream(self, mode: PyAudioStreamMode | None = None) -> pyaudio.Stream:
        if self._stream is None:
            raise PyAudioStreamError("Stream is not open")

        if mode is not None:
            supported = (
                self._config.mode.is_input
                if mode is PyAudioStreamMode.INPUT
                else self._config.mode.is_output
            )
            if not supported:
                raise PyAudioStreamError(
                    f"Incompatible stream mode: expected {mode.value}, "
                    f"got {self._config.mode.value}"
                )

        return self._stream
