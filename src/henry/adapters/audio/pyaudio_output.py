import asyncio
from types import TracebackType
from typing import Self

import numpy as np
import pyaudio
from loguru import logger

from ...domain import AudioFormat, AudioFrame
from ...ports import AudioOutput
from .pyaudio_session import PyAudioSession


class PyAudioOutput(AudioOutput):
    def __init__(
        self,
        session: PyAudioSession,
        frames_per_buffer: int,
    ) -> None:
        self._session = session
        self._frames_per_buffer = frames_per_buffer
        self._stream: pyaudio.Stream | None = None
        self._logger = logger.bind(component="PyAudioOutput")

    async def __aenter__(self) -> Self:
        pass
        if self._stream is not None:
            raise RuntimeError("Output stream is already open")

        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        stream = self._stream

        if stream is None:
            return

        self._stream = None

        await asyncio.to_thread(
            self._close_stream,
            stream,
        )

        self._logger.trace("Output stream CLOSED")

    async def play(self, frame: AudioFrame) -> None:
        samples = np.ascontiguousarray(
            frame.samples,
            dtype=np.float32,
        )

        raw_audio = samples.tobytes()

        stream = await self._get_stream(frame.format)

        await asyncio.to_thread(
            stream.write,
            raw_audio,
        )

    async def _get_stream(self, audio_format: AudioFormat) -> pyaudio.Stream:
        if self._stream is not None:
            return self._stream

        stream = await asyncio.to_thread(
            self._session.driver.open,
            rate=audio_format.sample_rate,
            channels=audio_format.channels,
            frames_per_buffer=self._frames_per_buffer,
            format=pyaudio.paFloat32,
            output=True,
        )

        self._stream = stream

        self._logger.trace(
            "Output stream OPENED: sample_rate={}, channels={}, frames_per_buffer={}",
            audio_format.sample_rate,
            audio_format.channels,
            self._frames_per_buffer,
        )

        return stream

    @staticmethod
    def _close_stream(stream: pyaudio.Stream) -> None:
        if stream.is_active():
            stream.stop_stream()

        stream.close()
