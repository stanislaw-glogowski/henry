import asyncio

from loguru import logger

from .audio import AudioChunk, AudioFrame
from .audio.service import AudioService
from .events import AppEventSink
from .speech.service import SpeechService


class Orchestrator:
    def __init__(
        self,
        audio: AudioService,
        speech: SpeechService,
        events: AppEventSink,
    ):
        self._audio = audio
        self._speech = speech

        self._events = events
        self._logger = logger.bind(component="Orchestrator")

    async def run(self, shutdown: asyncio.Event) -> None:
        audio_input: asyncio.Queue[AudioFrame] = asyncio.Queue()
        audio_output: asyncio.Queue[AudioFrame] = asyncio.Queue()
        audio_chunks: asyncio.Queue[AudioChunk] = asyncio.Queue()

        async with asyncio.TaskGroup() as tasks:
            self._logger.trace("Running tasks")

            audio_task = tasks.create_task(self._audio.run(audio_input, audio_output))
            speech_task = tasks.create_task(self._speech.run(audio_input, audio_chunks))

            await shutdown.wait()

            self._logger.trace("Cancelling tasks")

            audio_task.cancel()
            speech_task.cancel()
