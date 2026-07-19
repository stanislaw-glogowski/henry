from types import TracebackType
from typing import Self

import pyaudio
from loguru import logger


class PyAudioSession:
    def __init__(self) -> None:
        self._driver: pyaudio.PyAudio | None = None
        self._logger = logger.bind(component="PyAudioSession")

    @property
    def driver(self) -> pyaudio.PyAudio:
        if self._driver is None:
            raise RuntimeError("PyAudio session is not open")

        return self._driver

    def __enter__(self) -> Self:
        if self._driver is not None:
            raise RuntimeError("PyAudio session is already open")

        self._driver = pyaudio.PyAudio()

        self._logger.trace("Session OPENED")

        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._driver is not None:
            self._driver.terminate()
            self._driver = None

            self._logger.trace("Session TERMINATED")
