from ...domain import AudioDevices
from ...ports import AudioDriver
from .input import AVFAudioInput
from .output import AVFAudioOutput
from .process import AVFAudioProcess


class AVFAudioDriver(AudioDriver[AVFAudioInput, AVFAudioOutput]):
    """Own one native full-duplex process for capture and playback."""

    def __init__(self, process: AVFAudioProcess | None = None) -> None:
        super().__init__()
        self._process = process or AVFAudioProcess()
        self._input = AVFAudioInput(self._process)
        self._output = AVFAudioOutput(self._process)
        self._opened = False

    @property
    def input(self) -> AVFAudioInput:
        self._require_open()
        return self._input

    @property
    def output(self) -> AVFAudioOutput:
        self._require_open()
        return self._output

    @property
    def devices(self) -> AudioDevices:
        self._require_open()
        return self._process.devices

    def open(self) -> None:
        self._process.open()
        self._opened = True

    def close(self) -> None:
        try:
            self._process.close()
        finally:
            self._opened = False

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("AVFAudio driver is not open")
