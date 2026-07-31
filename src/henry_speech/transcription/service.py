import asyncio
import threading
from collections.abc import AsyncIterator

from henry_common.components import AbstractAsyncService

from ..audio import AudioFrame
from .domain import Transcription, TranscriptionChunk, TranscriptionText
from .ports import STTModel


class TranscriptionService(AbstractAsyncService):
    def __init__(
        self,
        stt_model: STTModel,
    ):
        super().__init__()
        self._stt_model = stt_model
        self._transcribe_cancel = threading.Event()
        self._transcribe_future: asyncio.Future[None] | None = None

    async def transcribe(self, frame: AudioFrame) -> AsyncIterator[Transcription]:
        if self._transcribe_future is not None:
            raise RuntimeError("Transcribe is already running")

        loop = asyncio.get_event_loop()
        responses = asyncio.Queue()

        self._transcribe_future = self._run_in_executor(
            self._run_transcribe,
            loop,
            frame,
            responses,
        )

        content = ""

        while True:
            response = await responses.get()
            try:
                match response:
                    case BaseException():
                        raise response
                    case None:
                        break
                    case TranscriptionChunk():
                        if response.content:
                            content += response.content
                            yield response
            finally:
                responses.task_done()

        if content:
            yield TranscriptionText(
                content=content,
            )
        else:
            yield None

        await self._cancel_transcribe()

    def _run_transcribe(
        self,
        loop: asyncio.AbstractEventLoop,
        frame: AudioFrame,
        responses: asyncio.Queue[TranscriptionChunk | BaseException | None],
    ) -> None:
        try:
            for chunk in self._stt_model.transcribe(frame):
                if self._transcribe_cancel.is_set():
                    break
                loop.call_soon_threadsafe(responses.put_nowait, chunk)
            loop.call_soon_threadsafe(responses.put_nowait, None)
        except BaseException as err:
            loop.call_soon_threadsafe(responses.put_nowait, err)

    async def _cancel_transcribe(self) -> None:
        if self._transcribe_future is None:
            return
        transcribe_future, self._transcribe_future = self._transcribe_future, None

        try:
            self._transcribe_cancel.set()
            await transcribe_future
        finally:
            self._transcribe_cancel.clear()
            self.transcribe_future = None

    def _open_resources(self) -> None:
        self._stt_model.open()

    def _close_resources(self) -> None:
        self._stt_model.close()

    async def _post_stop(self) -> None:
        await self._cancel_transcribe()
