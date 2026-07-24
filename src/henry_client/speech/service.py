import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Self

from loguru import logger

from ..audio import AudioFrame
from .domain import SpeechSegmenter
from .ports import STTModel, TTSModel


class SpeechServiceError(RuntimeError): ...


class SpeechService(AbstractAsyncContextManager):
    """Own STT and TTS model lifecycles in dedicated worker threads."""

    def __init__(
        self,
        stt_model: STTModel,
        tts_model: TTSModel,
    ) -> None:
        self._segmenter = SpeechSegmenter()

        self._stt_executor: ThreadPoolExecutor | None = None
        self._stt_model = stt_model

        self._tts_executor: ThreadPoolExecutor | None = None
        self._tts_model = tts_model

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

    def segment(
        self,
        frame: AudioFrame,
        speech_detected: bool,
    ) -> tuple[bool, AudioFrame | None]:
        """Feed one VAD-classified frame into the utterance segmenter."""
        return self._segmenter.feed(frame, speech_detected)

    async def transcribe(self, frame: AudioFrame) -> str | None:
        """Transcribe one complete utterance in the STT executor."""
        loop = asyncio.get_running_loop()
        executor = self._require_stt_executor()
        return await loop.run_in_executor(executor, self._stt_model.transcribe, frame)

    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        """Stream synthesized frames produced in the TTS executor."""
        loop = asyncio.get_running_loop()
        executor = self._require_tts_executor()
        responses: asyncio.Queue[AudioFrame | BaseException | None] = asyncio.Queue()
        job = loop.run_in_executor(
            executor,
            self._run_synthesis,
            text,
            loop,
            responses,
        )

        try:
            while True:
                response = await responses.get()
                try:
                    if response is None:
                        break
                    if isinstance(response, BaseException):
                        raise response
                    yield response
                finally:
                    responses.task_done()
        finally:
            await job

    def _run_synthesis(
        self,
        text: str,
        loop: asyncio.AbstractEventLoop,
        responses: asyncio.Queue[AudioFrame | BaseException | None],
    ) -> None:
        try:
            for frame in self._tts_model.synthesize(text):
                loop.call_soon_threadsafe(responses.put_nowait, frame)
            loop.call_soon_threadsafe(responses.put_nowait, None)
        except BaseException as error:
            loop.call_soon_threadsafe(responses.put_nowait, error)

    def _require_stt_executor(self) -> ThreadPoolExecutor:
        if self._stt_executor is None:
            raise SpeechServiceError("STT executor is not open")
        return self._stt_executor

    def _require_tts_executor(self) -> ThreadPoolExecutor:
        if self._tts_executor is None:
            raise SpeechServiceError("TTS executor is not open")
        return self._tts_executor

    async def _start(self) -> None:
        if self._stt_executor is not None:
            raise SpeechServiceError("STT executor is already started")
        if self._tts_executor is not None:
            raise SpeechServiceError("TTS executor is already started")

        loop = asyncio.get_running_loop()
        try:
            self._stt_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="SpeechService.stt",
            )
            await loop.run_in_executor(
                self._stt_executor,
                self._stt_model.open,
            )
            self._logger.debug("STT executor STARTED")

            self._tts_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="SpeechService.tts",
            )
            await loop.run_in_executor(
                self._tts_executor,
                self._tts_model.open,
            )
            self._logger.debug("TTS executor STARTED")
        except BaseException:
            await self._stop()
            raise

    async def _stop(self) -> None:
        try:
            await self._stop_stt()
        finally:
            await self._stop_tts()

    async def _stop_stt(self) -> None:
        executor = self._stt_executor
        if executor is None:
            return

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(executor, self._stt_model.close)
        finally:
            try:
                await asyncio.to_thread(executor.shutdown)
            finally:
                self._stt_executor = None
                self._logger.debug("STT executor STOPPED")

    async def _stop_tts(self) -> None:
        executor = self._tts_executor
        if executor is None:
            return

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(executor, self._tts_model.close)
        finally:
            try:
                await asyncio.to_thread(executor.shutdown)
            finally:
                self._tts_executor = None
                self._logger.debug("TTS executor STOPPED")
