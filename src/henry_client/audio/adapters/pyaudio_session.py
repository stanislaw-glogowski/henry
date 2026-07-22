from contextlib import AbstractContextManager
from types import TracebackType
from typing import Self

import pyaudio
from loguru import logger


class PyAudioSessionError(RuntimeError): ...


class PyAudioSession(AbstractContextManager):
    def __init__(self) -> None:
        self._driver: pyaudio.PyAudio | None = None
        self._logger = logger.bind(component="PyAudioSession")

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

    @property
    def driver(self) -> pyaudio.PyAudio:
        return self._require_driver()

    def _open(self) -> None:
        if self._driver is not None:
            raise PyAudioSessionError("Session is already open")

        self._driver = pyaudio.PyAudio()

        self._logger.debug("Session OPENED")

        assert self._driver is not None

        input_device = self._driver.get_default_input_device_info()
        output_device = self._driver.get_default_output_device_info()

        self._logger.debug(
            "Session OPENED: input_devic='{}', output_device='{}'",
            input_device.get("name"),
            output_device.get("name"),
        )

    def _close(self) -> None:
        if self._driver is None:
            return

        self._driver.terminate()
        self._driver = None

        self._logger.debug("Session TERMINATED")

    def _require_driver(self) -> pyaudio.PyAudio:
        if self._driver is None:
            raise PyAudioSessionError("Session is not open")

        return self._driver
