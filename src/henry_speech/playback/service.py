from henry_common import AbstractAsyncService

from ..audio import AudioFrame, AudioOutput


class PlaybackService(AbstractAsyncService):
    def __init__(
        self,
        audio_output: AudioOutput,
    ):
        super().__init__()
        self._audio_output = audio_output

    async def play(self, frame: AudioFrame) -> None:
        await self._run_in_executor(self._run_play, frame)

    def _run_play(self, frame: AudioFrame) -> None:
        self._audio_output.write(frame)

    def _open_resources(self) -> None:
        self._audio_output.open()

    def _close_resources(self) -> None:
        self._audio_output.close()
