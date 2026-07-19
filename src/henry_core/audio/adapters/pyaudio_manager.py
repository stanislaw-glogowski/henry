import pyaudio
from loguru import logger

from ..ports import StreamConfig, StreamManager, StreamManagerError
from .pyaudio_input import PyAudioInput
from .pyaudio_output import PyAudioOutput


class PyAudioManagerError(StreamManagerError): ...


class PyAudioManager(StreamManager):
    def __init__(self) -> None:
        self._driver: pyaudio.PyAudio | None = None
        self._logger = logger.bind(component="PyAudioManager")

    def open_input(self, config: StreamConfig) -> PyAudioInput:
        return PyAudioInput(self._require_driver(), config)

    def open_output(self, config: StreamConfig) -> PyAudioOutput:
        return PyAudioOutput(self._require_driver(), config)

    def open(self) -> None:
        if self._driver is not None:
            raise PyAudioManagerError("Driver is already open")

        self._driver = pyaudio.PyAudio()

        self._logger.trace("Driver OPENED")

    def close(self) -> None:
        if self._driver is None:
            return

        self._driver.terminate()
        self._driver = None

        self._logger.trace("Driver TERMINATED")

    def _require_driver(self) -> pyaudio.PyAudio:
        if self._driver is None:
            raise PyAudioManagerError("Driver is not open")

        return self._driver
