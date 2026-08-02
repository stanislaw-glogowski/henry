from collections.abc import Mapping

import pyaudio

from ...domain import AudioDevice, AudioDevices
from ...ports import AudioDriver
from .input import PyAudioInput
from .output import PyAudioOutput


class PyAudioDriver(AudioDriver[PyAudioInput, PyAudioOutput]):
    def __init__(self) -> None:
        super().__init__()
        self._session: pyaudio.PyAudio | None = None
        self._input: PyAudioInput | None = None
        self._output: PyAudioOutput | None = None
        self._devices: AudioDevices | None = None

    @property
    def input(self) -> PyAudioInput:
        input = self._input
        if input is None:
            raise RuntimeError("PyAudio driver is not open")
        return input

    @property
    def output(self) -> PyAudioOutput:
        output = self._output
        if output is None:
            raise RuntimeError("PyAudio driver is not open")
        return output

    @property
    def devices(self) -> AudioDevices:
        devices = self._devices
        if devices is None:
            raise RuntimeError("PyAudio driver is not open")
        return devices

    def open(self) -> None:
        if self._session is not None:
            raise RuntimeError("PyAudio session is already open")

        session = pyaudio.PyAudio()
        self._session = session
        try:
            input_device = session.get_default_input_device_info()
            output_device = session.get_default_output_device_info()
            self._devices = AudioDevices(
                input=self._build_device(input_device),
                output=self._build_device(output_device),
            )
            self._input = PyAudioInput(session=session)
            self._output = PyAudioOutput(session=session)
        except BaseException:
            self.close()
            raise

        self._logger.debug(
            "Session OPENED: input_device='{}', output_device='{}'",
            self.devices.input.name,
            self.devices.output.name,
        )

    def close(self) -> None:
        if self._session is None:
            return

        errors: list[Exception] = []

        if self._input:
            try:
                self._input.close()
            except Exception as err:
                errors.append(err)

        if self._output:
            try:
                self._output.close()
            except Exception as err:
                errors.append(err)

        try:
            self._session.terminate()
        except Exception as err:
            errors.append(err)

        self._input = None
        self._output = None
        self._devices = None
        self._session = None

        if errors:
            self._logger.warning("Session TERMINATED", errors=errors)
        else:
            self._logger.debug("Session TERMINATED")

    @staticmethod
    def _build_device(info: Mapping[str, float | int | str]) -> AudioDevice:
        name = info.get("name")
        identifier = info.get("index")
        return AudioDevice(
            name=name if isinstance(name, str) and name else "Unknown",
            identifier=str(identifier) if identifier is not None else None,
        )
