from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Self

import numpy as np
import pyaudio
from loguru import logger

from ..domain import AudioFormat, AudioFrame
from ..ports import InputStream, OutputStream
from .pyaudio_session import PyAudioSession

CHANNELS_NUM = 1

DUPLEX_SAMPLE_RATE = 16_000
DUPLEX_FRAMES_PER_BUFFER = 512

INPUT_SAMPLE_RATE = 16_000
INPUT_FRAMES_PER_BUFFER = 512

OUTPUT_SAMPLE_RATE = 22_050
OUTPUT_FRAMES_PER_BUFFER = 512


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
class PyAudioStreamConfig(AudioFormat):
    sample_rate: int = DUPLEX_SAMPLE_RATE
    frames_per_buffer: int = DUPLEX_FRAMES_PER_BUFFER
    channels: int = CHANNELS_NUM
    mode: PyAudioStreamMode = PyAudioStreamMode.DUPLEX


class PyAudioStreamError(RuntimeError): ...


class PyAudioStream(InputStream, OutputStream):
    @staticmethod
    def duplex(session: PyAudioSession) -> PyAudioStream:
        return PyAudioStream(
            session,
            PyAudioStreamConfig(),
        )

    @staticmethod
    def input(session: PyAudioSession) -> PyAudioStream:
        return PyAudioStream(
            session,
            PyAudioStreamConfig(
                sample_rate=INPUT_SAMPLE_RATE,
                frames_per_buffer=INPUT_FRAMES_PER_BUFFER,
                channels=CHANNELS_NUM,
                mode=PyAudioStreamMode.INPUT,
            ),
        )

    @staticmethod
    def output(session: PyAudioSession) -> PyAudioStream:
        return PyAudioStream(
            session,
            PyAudioStreamConfig(
                sample_rate=OUTPUT_SAMPLE_RATE,
                frames_per_buffer=OUTPUT_FRAMES_PER_BUFFER,
                channels=CHANNELS_NUM,
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

    def __enter__(self) -> Self:
        self._open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close()

    def read(self) -> AudioFrame:
        stream = self._require_stream(PyAudioStreamMode.INPUT)

        buffer = stream.read(
            self._config.frames_per_buffer,
            exception_on_overflow=False,
        )

        samples: np.ndarray = np.frombuffer(
            buffer,
            dtype=np.float32,
        )

        return self._config.build_frame(samples)

    def write(self, frame: AudioFrame) -> None:
        stream = self._require_stream(PyAudioStreamMode.OUTPUT)
        self._config.verify(frame)
        buffer = np.ascontiguousarray(
            frame.samples,
            dtype=np.float32,
        ).tobytes()

        stream.write(buffer)

    def _open(self) -> None:
        if self._stream is not None:
            raise PyAudioStreamError("Stream is already open")

        self._stream = self._session.driver.open(
            rate=self._config.sample_rate,
            channels=self._config.channels,
            frames_per_buffer=self._config.frames_per_buffer,
            input=self._config.mode.is_input,
            output=self._config.mode.is_output,
            format=pyaudio.paFloat32,
        )

        self._logger.debug(
            "Stream OPENED: sample_rate={}, channels={}, frames_per_buffer={}, mode={}",
            self._config.sample_rate,
            self._config.channels,
            self._config.frames_per_buffer,
            self._config.mode.value,
        )

    def _close(self) -> None:
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
            if mode.is_input != self._config.mode.is_input:
                raise PyAudioStreamError(
                    f"Incompatible stream mode: expected {mode.value}, "
                    f"got {self._config.mode.value}"
                )

        return self._stream
