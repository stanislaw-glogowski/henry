from ...domain import AudioFrame, AudioPlaybackOutcome
from ...ports import AudioOutput
from .process import AVFAudioProcess
from .protocol import PlaybackStatus


class AVFAudioOutput(AudioOutput):
    """Control playback through the shared native duplex process."""

    def __init__(self, process: AVFAudioProcess) -> None:
        super().__init__()
        self._process = process
        self._opened = False

    def write(self, frame: AudioFrame) -> AudioPlaybackOutcome:
        self._require_open()
        status = self._process.play(frame)
        if status is PlaybackStatus.INTERRUPTED:
            return AudioPlaybackOutcome.INTERRUPTED
        return AudioPlaybackOutcome.PLAYED

    def interrupt(self) -> None:
        self._require_open()
        self._process.interrupt()

    def duck(self) -> None:
        self._require_open()
        self._process.duck()

    def restore(self) -> None:
        self._require_open()
        self._process.restore()

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("AVFAudio output stream is already open")
        self._opened = True
        self._logger.debug("Stream OPENED")

    def close(self) -> None:
        if not self._opened:
            return
        self._opened = False
        self._logger.debug("Stream CLOSED")

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("AVFAudio output stream is not open")
