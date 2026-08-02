import asyncio
import threading
from collections.abc import AsyncIterator

from henry_common.components import AbstractAsyncService

from .domain import LanguageModelChunk, LanguageModelRequest, LanguageModelRole
from .ports import LanguageModel


class LanguageModelService(AbstractAsyncService):
    """Run all model operations in one model-owned worker thread."""

    def __init__(self, language_model: LanguageModel) -> None:
        super().__init__()
        self._language_model = language_model
        self._generation_cancel = threading.Event()
        self._generation_future: asyncio.Future[None] | None = None
        self._generation_lock = asyncio.Lock()

    async def prepare(self, role: LanguageModelRole) -> None:
        async with self._generation_lock:
            await self._run_in_executor(self._language_model.prepare, role)

    async def generate(
        self,
        request: LanguageModelRequest,
    ) -> AsyncIterator[LanguageModelChunk]:
        async with self._generation_lock:
            if self._generation_future is not None:
                raise RuntimeError("Language model generation is already in progress")

            loop = asyncio.get_running_loop()
            responses: asyncio.Queue[LanguageModelChunk | BaseException | None] = (
                asyncio.Queue()
            )
            self._generation_future = self._run_in_executor(
                self._run_generate,
                loop,
                request,
                responses,
            )

            try:
                while True:
                    response = await responses.get()
                    try:
                        match response:
                            case BaseException():
                                raise response
                            case None:
                                break
                            case LanguageModelChunk():
                                if response.content:
                                    yield response
                    finally:
                        responses.task_done()
            finally:
                await self._cancel_generation()

    def _run_generate(
        self,
        loop: asyncio.AbstractEventLoop,
        request: LanguageModelRequest,
        responses: asyncio.Queue[LanguageModelChunk | BaseException | None],
    ) -> None:
        try:
            for chunk in self._language_model.generate(request):
                if self._generation_cancel.is_set():
                    break
                loop.call_soon_threadsafe(responses.put_nowait, chunk)
            loop.call_soon_threadsafe(responses.put_nowait, None)
        except BaseException as error:
            loop.call_soon_threadsafe(responses.put_nowait, error)

    async def _cancel_generation(self) -> None:
        if self._generation_future is None:
            return
        future, self._generation_future = self._generation_future, None
        try:
            self._generation_cancel.set()
            await future
        finally:
            self._generation_cancel.clear()

    def _open_resources(self) -> None:
        self._language_model.open()

    def _close_resources(self) -> None:
        self._language_model.close()

    async def _post_stop(self) -> None:
        await self._cancel_generation()
