import asyncio
import queue
import threading
from collections.abc import AsyncIterator
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

from .domain import AudioChunk, AudioFrame
from .ports import InputStream, OutputStream, VADModel

type InputRequest = DoRead | None
type OutputRequest = DoWrite | None

VAD_THRESHOLD = 0.5

INPUT_THREAD_NAME = "AudioService.input_worker"
OUTPUT_THREAD_NAME = "AudioService.output_worker"


@dataclass(frozen=True, slots=True)
class DoRead:
    response: asyncio.Queue[AudioChunk | BaseException | None]


@dataclass(frozen=True, slots=True)
class DoWrite:
    frame: AudioFrame
    response: asyncio.Future[None]


class AudioServiceError(RuntimeError): ...


class AudioService(AbstractAsyncContextManager):
    def __init__(
        self,
        input_stream: InputStream,
        output_stream: OutputStream,
        vad_model: VADModel,
        vad_threshold=VAD_THRESHOLD,
    ) -> None:
        self._input_stream = input_stream
        self._input_cancel = threading.Event()
        self._input_thread: threading.Thread | None = None
        self._input_requests: queue.Queue[InputRequest] = queue.Queue()

        self._output_stream = output_stream
        self._output_thread: threading.Thread | None = None
        self._output_requests: queue.Queue[OutputRequest] = queue.Queue()

        self._vad_model = vad_model
        self._vad_threshold = vad_threshold
        self._logger = logger.bind(component="AudioService")

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

    async def read(self) -> AsyncIterator[AudioChunk]:
        request = DoRead(
            response=asyncio.Queue(),
        )
        self._input_requests.put_nowait(request)

        while True:
            frame = await request.response.get()
            if frame is None:
                return
            if isinstance(frame, BaseException):
                raise frame
            yield frame
            request.response.task_done()

    async def write(self, frame: AudioFrame) -> None:
        request = DoWrite(
            frame=frame,
            response=asyncio.Future(),
        )
        self._output_requests.put_nowait(request)
        await request.response

    async def _start(self) -> None:
        if self._input_thread is not None or self._output_thread is not None:
            raise AudioServiceError("Workers already started")

        loop = asyncio.get_running_loop()

        input_ready = loop.create_future()
        self._input_cancel.clear()
        self._input_requests = queue.Queue()

        output_ready = loop.create_future()
        self._output_requests = queue.Queue()

        try:
            self._input_thread = threading.Thread(
                target=self._input_worker,
                args=(loop, input_ready),
                name=INPUT_THREAD_NAME,
            )
            assert self._input_thread is not None
            self._input_thread.start()
            await input_ready
            self._logger.debug("Input worker READY")

            self._output_thread = threading.Thread(
                target=self._output_worker,
                args=(loop, output_ready),
                name=OUTPUT_THREAD_NAME,
            )
            assert self._output_thread is not None
            self._output_thread.start()
            await output_ready
            self._logger.debug("Output worker READY")
        except BaseException:
            await self._stop()
            raise

    async def _stop(self) -> None:
        if self._input_thread is not None:
            self._input_cancel.set()
            self._input_requests.put_nowait(None)
            await join_started_thread(self._input_thread)
            self._input_thread = None
            self._logger.debug("Input worker STOPPED")

        if self._output_thread is not None:
            self._output_requests.put_nowait(None)
            await join_started_thread(self._output_thread)
            self._output_thread = None
            self._logger.debug("Output worker STOPPED")

    def _input_worker(
        self,
        loop: asyncio.AbstractEventLoop,
        ready: asyncio.Future[None],
    ) -> None:
        try:
            with (
                self._input_stream as stream,
                self._vad_model as vad_model,
            ):
                loop.call_soon_threadsafe(
                    set_future_result_if_pending,
                    ready,
                    None,
                )

                while True:
                    request = self._input_requests.get()

                    if request is None:
                        self._input_requests.task_done()
                        return

                    try:
                        while True:
                            if self._input_cancel.is_set():
                                loop.call_soon_threadsafe(
                                    request.response.put_nowait,
                                    None,
                                )
                                break

                            frame = stream.read()
                            vad_score = vad_model.predict(frame)
                            is_speech = vad_score >= self._vad_threshold

                            loop.call_soon_threadsafe(
                                request.response.put_nowait,
                                frame.build_chunk(is_speech, vad_score),
                            )
                    finally:
                        self._input_requests.task_done()
        except BaseException as error:
            loop.call_soon_threadsafe(
                set_future_exception_if_pending,
                ready,
                error,
            )

    def _output_worker(
        self,
        loop: asyncio.AbstractEventLoop,
        ready: asyncio.Future[None],
    ) -> None:
        request: OutputRequest = None

        try:
            with self._output_stream as stream:
                loop.call_soon_threadsafe(
                    set_future_result_if_pending,
                    ready,
                    None,
                )

                while True:
                    request = self._output_requests.get()

                    if request is None:
                        self._output_requests.task_done()
                        break

                    try:
                        stream.write(request.frame)

                        loop.call_soon_threadsafe(
                            set_future_result_if_pending,
                            request.response,
                            None,
                        )
                    finally:
                        self._output_requests.task_done()
        except BaseException as err:
            loop.call_soon_threadsafe(
                set_future_exception_if_pending,
                ready,
                err,
            )
            if request is not None:
                loop.call_soon_threadsafe(
                    set_future_exception_if_pending,
                    request.response,
                    err,
                )
