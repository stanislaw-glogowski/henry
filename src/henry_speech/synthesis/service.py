import asyncio
import threading
from collections.abc import AsyncIterator

from henry_common import AbstractAsyncService

from ..audio import AudioFrame
from .ports import SynthesisModel


class SynthesisService(AbstractAsyncService):
    def __init__(
        self,
        model: SynthesisModel,
    ):
        super().__init__()
        self._model = model
        self._synthesize_cancel = threading.Event()
        self._synthesize_future: asyncio.Future[None] | None = None

    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        if self._synthesize_future is not None:
            raise RuntimeError("Synthesize is already running")

        loop = asyncio.get_event_loop()
        responses = asyncio.Queue()

        self._synthesize_future = self._run_in_executor(
            self._run_synthesize,
            loop,
            text,
            responses,
        )

        while True:
            response = await responses.get()
            try:
                match response:
                    case BaseException():
                        raise response
                    case None:
                        break
                    case AudioFrame():
                        yield response
            finally:
                responses.task_done()

        await self._cancel_synthesize()

    def _run_synthesize(
        self,
        loop: asyncio.AbstractEventLoop,
        text: str,
        responses: asyncio.Queue[AudioFrame | BaseException | None],
    ) -> None:
        try:
            frames = self._model.synthesize(text)

            for frame in frames:
                if self._synthesize_cancel.is_set():
                    break
                loop.call_soon_threadsafe(responses.put_nowait, frame)
            loop.call_soon_threadsafe(responses.put_nowait, None)
        except BaseException as err:
            loop.call_soon_threadsafe(responses.put_nowait, err)

    async def _cancel_synthesize(self) -> None:
        if self._synthesize_future is None:
            return
        synthesize_future, self._synthesize_future = self._synthesize_future, None

        try:
            self._synthesize_cancel.set()
            await synthesize_future
        finally:
            self._synthesize_cancel.clear()
            self.synthesize_future = None

    def _open_resources(self) -> None:
        self._model.open()

    def _close_resources(self) -> None:
        self._model.close()

    async def _post_stop(self) -> None:
        await self._cancel_synthesize()
