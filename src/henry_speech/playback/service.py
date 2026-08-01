import asyncio

from henry_common.components import AbstractAsyncService

from ..audio import AudioFrame, AudioOutput, AudioPlaybackOutcome


class PlaybackService(AbstractAsyncService):
    def __init__(
        self,
        audio_output: AudioOutput,
    ):
        super().__init__()
        self._audio_output = audio_output

    async def play(self, frame: AudioFrame) -> AudioPlaybackOutcome:
        return await self._run_in_executor(self._run_play, frame)

    async def interrupt(self) -> None:
        # Playback may occupy the service executor, so interruption must use a
        # separate thread to reach the device concurrently.
        await asyncio.to_thread(self._audio_output.interrupt)

    async def duck(self) -> None:
        await asyncio.to_thread(self._audio_output.duck)

    async def restore(self) -> None:
        await asyncio.to_thread(self._audio_output.restore)

    def _run_play(self, frame: AudioFrame) -> AudioPlaybackOutcome:
        return self._audio_output.write(frame)

    def _open_resources(self) -> None:
        self._audio_output.open()

    def _close_resources(self) -> None:
        self._audio_output.close()
