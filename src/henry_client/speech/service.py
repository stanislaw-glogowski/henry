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

from ..audio import AudioChunk, AudioFrame
from .domain import SpeechSegmenter
from .ports import STTModel, TTSModel

STT_THREAD_NAME = "SpeechService.stt_worker"
TTS_THREAD_NAME = "SpeechService.tts_worker"


type STTRequest = DoTranscribe | None
type TTSRequest = DoSynthesize | None


@dataclass(frozen=True, slots=True)
class DoTranscribe:
    frame: AudioFrame
    response: asyncio.Future[str | None]


@dataclass(frozen=True, slots=True)
class DoSynthesize:
    text: str
    response: asyncio.Queue[AudioFrame | BaseException | None]


class SpeechServiceError(RuntimeError): ...


class SpeechService(AbstractAsyncContextManager):
    def __init__(
        self,
        stt_model: STTModel,
        tts_model: TTSModel,
    ) -> None:
        self._segmenter = SpeechSegmenter()
        self._stt_model = stt_model
        self._stt_thread: threading.Thread | None = None
        self._stt_requests: queue.Queue[STTRequest] = queue.Queue()

        self._tts_model = tts_model
        self._tts_thread: threading.Thread | None = None
        self._tts_requests: queue.Queue[TTSRequest] = queue.Queue()

        self._logger = logger.bind(component="SpeechService")

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

    def detect(self, chunk: AudioChunk) -> tuple[bool, AudioFrame | None]:
        return self._segmenter.feed(chunk)

    async def transcribe(self, frame: AudioFrame) -> str | None:
        request = DoTranscribe(
            frame=frame,
            response=asyncio.Future(),
        )
        self._stt_requests.put_nowait(request)
        return await request.response

    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        request = DoSynthesize(
            text=text,
            response=asyncio.Queue(),
        )
        self._tts_requests.put_nowait(request)

        while True:
            frame = await request.response.get()
            if frame is None:
                return
            if isinstance(frame, BaseException):
                raise frame
            yield frame
            request.response.task_done()

    async def _start(self) -> None:
        if self._stt_thread is not None or self._stt_thread is not None:
            raise SpeechServiceError("Workers already started")

        loop = asyncio.get_running_loop()

        stt_ready = loop.create_future()
        self._stt_requests = queue.Queue()

        tts_ready = loop.create_future()
        self._tts_requests = queue.Queue()

        try:
            self._stt_thread = threading.Thread(
                target=self._stt_worker,
                args=(loop, stt_ready),
                name=STT_THREAD_NAME,
            )

            assert self._stt_thread is not None
            self._stt_thread.start()
            await stt_ready
            self._logger.debug("STT worker READY")

            self._tts_thread = threading.Thread(
                target=self._tts_worker,
                args=(loop, tts_ready),
                name=TTS_THREAD_NAME,
            )

            assert self._tts_thread is not None
            self._tts_thread.start()
            await tts_ready
            self._logger.debug("TTS worker READY")
        except BaseException:
            await self._stop()
            raise

    async def _stop(self) -> None:
        if self._stt_thread is not None:
            self._stt_requests.put_nowait(None)
            await join_started_thread(self._stt_thread)
            self._stt_thread = None
            self._logger.debug("STT worker STOPPED")

        if self._tts_thread is not None:
            self._tts_requests.put_nowait(None)
            await join_started_thread(self._tts_thread)
            self._tts_thread = None
            self._logger.debug("TTS worker STOPPED")

    def _stt_worker(
        self,
        loop: asyncio.AbstractEventLoop,
        ready: asyncio.Future[None],
    ) -> None:
        request: STTRequest | None = None

        try:
            with self._stt_model as model:
                loop.call_soon_threadsafe(
                    set_future_result_if_pending,
                    ready,
                    None,
                )

                while True:
                    request = self._stt_requests.get()

                    if request is None:
                        self._stt_requests.task_done()
                        break

                    try:
                        text = model.transcribe(
                            request.frame,
                        )

                        loop.call_soon_threadsafe(
                            set_future_result_if_pending,
                            request.response,
                            text,
                        )
                    finally:
                        self._stt_requests.task_done()
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

    def _tts_worker(
        self,
        loop: asyncio.AbstractEventLoop,
        ready: asyncio.Future[None],
    ) -> None:
        request: TTSRequest | None = None

        try:
            with self._tts_model as model:
                loop.call_soon_threadsafe(
                    set_future_result_if_pending,
                    ready,
                    None,
                )

                while True:
                    request = self._tts_requests.get()

                    if request is None:
                        self._tts_requests.task_done()
                        break

                    try:
                        frames = model.synthesize(
                            request.text,
                        )

                        for frame in frames:
                            loop.call_soon_threadsafe(
                                request.response.put_nowait,
                                frame,
                            )

                        loop.call_soon_threadsafe(
                            request.response.put_nowait,
                            None,
                        )
                    finally:
                        self._tts_requests.task_done()
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
