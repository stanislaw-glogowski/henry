import pyaudio

from ...domain import AudioFormat, AudioFrame
from ...ports import AudioInput


class PyAudioInput(AudioInput):
    _FORMAT = AudioFormat(
        sample_rate=16_000,
        channels=1,
    )
    _FRAMES_PER_BUFFER = 512

    def __init__(
        self,
        session: pyaudio.PyAudio,
    ) -> None:
        super().__init__()
        self._session = session
        self._stream: pyaudio.Stream | None = None

    def read(self) -> AudioFrame:
        if self._stream is None:
            raise RuntimeError("PyAudio input stream is not open")

        buffer = self._stream.read(
            self._FRAMES_PER_BUFFER,
            exception_on_overflow=False,
        )
        return self._FORMAT.build_frame(buffer)

    def open(self) -> None:
        if self._stream is not None:
            raise RuntimeError("PyAudio input stream is already open")

        self._stream = self._session.open(
            rate=self._FORMAT.sample_rate,
            channels=self._FORMAT.channels,
            frames_per_buffer=self._FRAMES_PER_BUFFER,
            input=True,
            format=pyaudio.paFloat32,
        )
        self._logger.debug(
            "Stream OPENED: sample_rate={}, channels={}, frames_per_buffer={}",
            self._FORMAT.sample_rate,
            self._FORMAT.channels,
            self._FRAMES_PER_BUFFER,
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
