import pyaudio

from ...domain import AudioFormat, AudioFrame
from ...ports import AudioOutput


class PyAudioOutput(AudioOutput):
    def __init__(
        self,
        session: pyaudio.PyAudio,
        frames_per_buffer: int,
    ) -> None:
        super().__init__()
        self._session = session
        self._frames_per_buffer = frames_per_buffer
        self._format: AudioFormat | None = None
        self._stream: pyaudio.Stream | None = None

    def write(self, frame: AudioFrame) -> None:
        stream = self._require_stream(frame.format)
        stream.write(frame.to_bytes())

    def open(self) -> None:
        if self._stream is not None:
            raise RuntimeError("Stream is already open")

    def close(self) -> None:
        stream = self._stream

        if stream is None:
            return

        if stream.is_active():
            stream.stop_stream()

        stream.close()

        self._stream = None
        self._format = None
        self._logger.debug("Stream CLOSED")

    def _require_stream(self, format: AudioFormat) -> pyaudio.Stream:
        if self._stream is not None and self._format is not None:
            self._format.verify(format)
            return self._stream

        stream = self._session.open(
            rate=format.sample_rate,
            channels=format.channels,
            frames_per_buffer=self._frames_per_buffer,
            output=True,
            format=pyaudio.paFloat32,
        )

        self._stream = stream
        self._format = format
        self._logger.debug(
            "Stream OPENED: sample_rate={}, channels={}, frames_per_buffer={}",
            format.sample_rate,
            format.channels,
            self._frames_per_buffer,
        )
        return stream
