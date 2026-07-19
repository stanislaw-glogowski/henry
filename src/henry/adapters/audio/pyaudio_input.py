import asyncio
from types import TracebackType
from typing import Self

import numpy as np
import pyaudio
from loguru import logger

from ...domain import AudioFormat, AudioFrame
from ...ports import AudioInput
from .pyaudio_session import PyAudioSession


class PyAudioInput(AudioInput):
    def __init__(
        self,
        session: PyAudioSession,
        audio_format: AudioFormat,
        frames_per_buffer: int,
    ) -> None:
        self._session = session
        self._format = audio_format
        self._frames_per_buffer = frames_per_buffer
        self._stream: pyaudio.Stream | None = None
        self._logger = logger.bind(component="PyAudioInput")

    async def __aenter__(self) -> Self:
        if self._stream is not None:
            raise RuntimeError("Input stream is already open")

        self._stream = self._session.driver.open(
            rate=self._format.sample_rate,
            channels=self._format.channels,
            frames_per_buffer=self._frames_per_buffer,
            format=pyaudio.paFloat32,
            input=True,
        )

        self._logger.trace(
            "Input stream OPENED: sample_rate={}, channels={}, frames_per_buffer={}",
            self._format.sample_rate,
            self._format.channels,
            self._frames_per_buffer,
        )

        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

        self._logger.trace("Input stream CLOSED")

    async def read(self) -> AudioFrame:
        stream = self._require_stream()

        raw_audio = await asyncio.to_thread(
            stream.read,
            self._frames_per_buffer,
            exception_on_overflow=False,
        )

        samples: np.ndarray = np.frombuffer(
            raw_audio,
            dtype=np.float32,
        )

        return AudioFrame(
            samples=samples,
            format=self._format,
        )

    async def close(self) -> None:
        stream = self._stream

        if stream is None:
            return

        self._stream = None

        await asyncio.to_thread(
            self._close_stream,
            stream,
        )

    def _require_stream(self) -> pyaudio.Stream:
        if self._stream is None:
            raise RuntimeError("Input stream is not open")

        return self._stream

    @staticmethod
    def _close_stream(stream: pyaudio.Stream) -> None:
        if stream.is_active():
            stream.stop_stream()

        stream.close()
