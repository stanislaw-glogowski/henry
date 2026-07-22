import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from loguru import logger

from henry_common.concurrency import (
    join_started_thread,
    set_future_exception_if_pending,
    set_future_result_if_pending,
)

from .domain import (
    ConversationStore,
    Message,
    MessageChunk,
    MessageLine,
    MessageReply,
    MessageRole,
)
from .ports import LanguageModel

type ModelRequest = DoGenerate | None


@dataclass(frozen=True, slots=True)
class DoGenerate:
    messages: Sequence[Message]
    response: asyncio.Queue[MessageChunk | BaseException | None]


class ConversationServiceError(RuntimeError): ...


class ConversationService(AbstractAsyncContextManager):
    _MODEL_THREAD_NAME = "ConversationService.model_worker"

    def __init__(
        self,
        model: LanguageModel,
        system_prompt: str | None = None,
    ) -> None:
        self._store = ConversationStore()

        self._model = model
        self._model_thread: threading.Thread | None = None
        self._model_requests: queue.Queue[ModelRequest] = queue.Queue()

        self._system_prompt = system_prompt
        self._logger = logger.bind(component="ConversationService")

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

    async def generate_reply(self, text: str) -> AsyncIterator[MessageReply]:
        self._store.add_user_message(text)

        messages = self._store.messages

        if self._system_prompt is not None:
            messages = (
                Message(role=MessageRole.SYSTEM, content=self._system_prompt),
                *messages,
            )

        request = DoGenerate(
            messages=messages,
            response=asyncio.Queue(),
        )
        self._model_requests.put_nowait(request)

        content = ""
        buffer = ""
        while True:
            chunk = await request.response.get()
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
                    yield MessageLine(line)

            request.response.task_done()

        self._store.add_assistant_message(content)

        if buffer:
            yield MessageLine(buffer)

        if content:
            yield content
        else:
            yield None

    async def _start(self) -> None:
        if self._model_thread is not None:
            raise ConversationServiceError("Model worker already started")

        loop = asyncio.get_running_loop()

        model_ready = loop.create_future()
        self._model_requests = queue.Queue()

        try:
            self._model_thread = threading.Thread(
                target=self._model_worker,
                args=(loop, model_ready),
                name=self._MODEL_THREAD_NAME,
            )
            assert self._model_thread is not None
            self._model_thread.start()
            await model_ready
            self._logger.debug("Model worker READY")
        except BaseException:
            await self._stop()
            raise

    async def _stop(self) -> None:
        self._store.reset()

        if self._model_thread is not None:
            self._model_requests.put_nowait(None)
            await join_started_thread(self._model_thread)
            self._model_thread = None
            self._logger.debug("Model worker STOPPED")

    def _model_worker(
        self,
        loop: asyncio.AbstractEventLoop,
        ready: asyncio.Future[None],
    ) -> None:
        try:
            with self._model as model:
                loop.call_soon_threadsafe(
                    set_future_result_if_pending,
                    ready,
                    None,
                )

                while True:
                    request = self._model_requests.get()

                    if request is None:
                        self._model_requests.task_done()
                        return

                    try:
                        chunks = model.generate(request.messages)

                        for chunk in chunks:
                            loop.call_soon_threadsafe(
                                request.response.put_nowait,
                                chunk,
                            )

                        loop.call_soon_threadsafe(
                            request.response.put_nowait,
                            None,
                        )
                    finally:
                        self._model_requests.task_done()
        except BaseException as err:
            loop.call_soon_threadsafe(
                set_future_exception_if_pending,
                ready,
                err,
            )
            if request is not None:
                loop.call_soon_threadsafe(
                    request.response.put_nowait,
                    err,
                )
