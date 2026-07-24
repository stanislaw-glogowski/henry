import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Self

from loguru import logger

from .domain import (
    Reply,
    ReplyChunk,
    ReplyLine,
    ReplyRequest,
    ReplyText,
)
from .ports import Responder


class ReplyServiceError(RuntimeError): ...


class ReplyService(AbstractAsyncContextManager):
    """Own the blocking responder lifecycle in a single worker thread."""

    def __init__(
        self,
        responder: Responder,
    ) -> None:
        self._responder = responder
        self._replying = asyncio.Event()
        self._executor: ThreadPoolExecutor | None = None
        self._logger = logger.bind(component="ReplyService")

    async def __aenter__(self) -> Self:
        await self._start()
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._stop()

    async def reply(self, request: ReplyRequest) -> AsyncIterator[Reply]:
        """Stream raw chunks, complete lines, and the accumulated reply text."""
        if self._replying.is_set():
            return

        if self._executor is None:
            raise ReplyServiceError("Reply executor is not open")

        loop = asyncio.get_running_loop()
        responses: asyncio.Queue[ReplyChunk | BaseException | None] = asyncio.Queue()

        self._replying.set()

        job = loop.run_in_executor(
            self._executor,
            self._run_reply,
            request,
            loop,
            responses,
        )

        content = ""
        buffer = ""
        try:
            while True:
                chunk = await responses.get()
                try:
                    if chunk is None:
                        break
                    if isinstance(chunk, BaseException):
                        raise chunk

                    content += chunk.content
                    buffer += chunk.content

                    yield chunk

                    while "\n" in buffer:
                        line, _, buffer = buffer.partition("\n")

                        if line:
                            yield ReplyLine(line)

                finally:
                    responses.task_done()

            if buffer:
                yield ReplyLine(buffer)

            yield ReplyText(content)
        finally:
            await job
            self._replying.clear()

    def _run_reply(
        self,
        request: ReplyRequest,
        loop: asyncio.AbstractEventLoop,
        responses: asyncio.Queue[ReplyChunk | BaseException | None],
    ) -> None:
        try:
            chunks = self._responder.respond(request)

            for chunk in chunks:
                loop.call_soon_threadsafe(
                    responses.put_nowait,
                    chunk,
                )
            loop.call_soon_threadsafe(
                responses.put_nowait,
                None,
            )
        except BaseException as err:
            loop.call_soon_threadsafe(responses.put_nowait, err)

    async def _start(self) -> None:
        if self._executor is not None:
            raise ReplyServiceError("Reply executor is already started")

        loop = asyncio.get_running_loop()

        try:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ReplyService",
            )
            await loop.run_in_executor(self._executor, self._responder.open)
            self._logger.debug("Reply executor STARTED")
        except BaseException:
            await self._stop()
            raise

    async def _stop(self) -> None:
        if self._executor is None:
            return

        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(self._executor, self._responder.close)
        finally:
            try:
                await asyncio.to_thread(self._executor.shutdown)
            finally:
                self._executor = None
                self._logger.debug("Reply executor STOPPED")
