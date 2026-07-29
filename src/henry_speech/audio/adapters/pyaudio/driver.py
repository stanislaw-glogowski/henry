import pyaudio

from ...domain import AudioFormat
from ...ports import AudioDriver
from .input import PyAudioInput
from .output import PyAudioOutput


class PyAudioDriver(AudioDriver[PyAudioInput, PyAudioOutput]):
    _INPUT_FORMAT = AudioFormat(
        sample_rate=16_000,
        channels=1,
    )
    _INPUT_FRAMES_PER_BUFFER = 512
    _OUTPUT_FRAMES_PER_BUFFER = 512

    def __init__(self) -> None:
        super().__init__()
        self._session: pyaudio.PyAudio | None = None
        self._input: PyAudioInput | None = None
        self._output: PyAudioOutput | None = None

    def get_input(self) -> PyAudioInput:
        input = self._input
        if input is None:
            input = PyAudioInput(
                session=self._require_session(),
                format=self._INPUT_FORMAT,
                frames_per_buffer=self._INPUT_FRAMES_PER_BUFFER,
            )
            self._input = input
        return input

    def get_output(self) -> PyAudioOutput:
        output = self._output
        if output is None:
            output = PyAudioOutput(
                session=self._require_session(),
                frames_per_buffer=self._OUTPUT_FRAMES_PER_BUFFER,
            )
            self._output = output
        return output

    def open(self) -> None:
        if self._session is not None:
            raise RuntimeError("Session is already open")

        self._session = pyaudio.PyAudio()

        self._logger.debug("Session OPENED")

        assert self._session is not None

        input_device = self._session.get_default_input_device_info()
        output_device = self._session.get_default_output_device_info()

        self._logger.debug(
            "Session OPENED: input_device='{}', output_device='{}'",
            input_device.get("name"),
            output_device.get("name"),
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
        self._session = None

        if errors:
            self._logger.warn("Session TERMINATED: errors:{}", errors)
        else:
            self._logger.debug("Session TERMINATED")

    def _require_session(self) -> pyaudio.PyAudio:
        if self._session is None:
            raise RuntimeError("Session is not open")
        return self._session
