import pyaudio

from ...domain import AudioFormat, AudioFrame
from ...ports import AudioInput


class PyAudioInput(AudioInput):
    def __init__(
        self,
        session: pyaudio.PyAudio,
        format: AudioFormat,
        frames_per_buffer: int = 512,
    ) -> None:
        super().__init__()
        self._session = session
        self._format = format
        self._frames_per_buffer = frames_per_buffer
        self._stream: pyaudio.Stream | None = None

    def read(self) -> AudioFrame:
        if self._stream is None:
            raise RuntimeError("Stream is is not open")

        buffer = self._stream.read(
            self._frames_per_buffer,
            exception_on_overflow=False,
        )
        return self._format.build_frame(buffer)

    def open(self) -> None:
        if self._stream is not None:
            raise RuntimeError("Stream is already open")

        self._stream = self._session.open(
            rate=self._format.sample_rate,
            channels=self._format.channels,
            frames_per_buffer=self._frames_per_buffer,
            input=True,
            format=pyaudio.paFloat32,
        )
        self._logger.debug(
            "Stream OPENED: sample_rate={}, channels={}, frames_per_buffer={}",
            self._format.sample_rate,
            self._format.channels,
            self._frames_per_buffer,
        )

    def close(self) -> None:
        stream = self._stream

        if stream is None:
            return

        if stream.is_active():
            stream.stop_stream()

        stream.close()

        self._stream = None
        self._logger.debug("Stream CLOSED")
