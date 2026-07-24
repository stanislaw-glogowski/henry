import asyncio
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager, ExitStack
from types import TracebackType
from typing import Self

from loguru import logger

from .domain import AudioChunk, AudioFrame
from .ports import InputStream, OutputStream, VADModel, WakeWordModel


class AudioServiceError(RuntimeError): ...


class AudioService(AbstractAsyncContextManager):
    """Own audio input and output resources in dedicated worker threads."""

    def __init__(
        self,
        input_stream: InputStream,
        output_stream: OutputStream,
        vad_model: VADModel,
        wakeword_model: WakeWordModel,
    ) -> None:
        self._capture_future: asyncio.Future[None] | None = None
        self._capture_executor: ThreadPoolExecutor | None = None
        self._capture_resources: ExitStack | None = None
        self._playback_executor: ThreadPoolExecutor | None = None

        self._input_cancel = threading.Event()
        self._input_stream = input_stream
        self._output_stream = output_stream
        self._vad_model = vad_model
        self._wakeword_enabled = threading.Event()
        self._wakeword_reset = threading.Event()
        self._wakeword_model = wakeword_model

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

    def enable_wakeword(self) -> None:
        """Arm wake-word detection."""
        self._wakeword_enabled.set()

    def disable_wakeword(self) -> None:
        """Disarm wake-word detection."""
        self._wakeword_enabled.clear()

    def reset_wakeword(self) -> None:
        """Request a wake-word model reset in the capture worker."""
        self._wakeword_reset.set()

    async def capture(self) -> AsyncIterator[AudioChunk]:
        """Stream microphone frames enriched with VAD and wake-word results."""
        if self._capture_future is not None:
            raise AudioServiceError("Capture is already running")

        loop = asyncio.get_running_loop()
        executor = self._require_capture_executor()
        responses: asyncio.Queue[AudioChunk | BaseException | None] = asyncio.Queue()

        self._capture_future = loop.run_in_executor(
            executor,
            self._run_capture_loop,
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
            self._input_cancel.set()
            assert self._capture_future is not None
            await self._capture_future

    async def playback(self, frame: AudioFrame) -> None:
        """Write one frame in the output stream's worker thread."""
        loop = asyncio.get_running_loop()
        executor = self._require_playback_executor()

        await loop.run_in_executor(
            executor,
            self._output_stream.write,
            frame,
        )

    def _run_capture_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        responses: asyncio.Queue[AudioChunk | BaseException | None],
    ) -> None:
        sequence_id = 0
        try:
            while not self._input_cancel.is_set():
                sequence_id += 1

                frame = self._input_stream.read()
                vad_score = self._vad_model.predict(frame)
                wakeword_score: float | None = None

                if self._wakeword_reset.is_set():
                    self._wakeword_model.reset()
                    self._wakeword_reset.clear()

                if self._wakeword_enabled.is_set():
                    wakeword_score = self._wakeword_model.predict(frame)

                loop.call_soon_threadsafe(
                    responses.put_nowait,
                    AudioChunk(
                        sequence_id=sequence_id,
                        frame=frame,
                        vad_score=vad_score,
                        wakeword_score=wakeword_score,
                    ),
                )
            loop.call_soon_threadsafe(
                responses.put_nowait,
                None,
            )
        except BaseException as err:
            loop.call_soon_threadsafe(
                responses.put_nowait,
                err,
            )

    def _require_capture_executor(self) -> ThreadPoolExecutor:
        if self._capture_executor is None:
            raise AudioServiceError("Capture executor is not open")

        return self._capture_executor

    def _require_playback_executor(self) -> ThreadPoolExecutor:
        if self._playback_executor is None:
            raise AudioServiceError("Playback executor is not open")

        return self._playback_executor

    async def _start(self) -> None:
        if self._capture_executor is not None:
            raise AudioServiceError("Capture executor is already started")

        if self._playback_executor is not None:
            raise AudioServiceError("Playback executor is already started")

        loop = asyncio.get_running_loop()

        try:
            self._input_cancel.clear()
            self._wakeword_enabled.set()
            self._wakeword_reset.clear()

            self._capture_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="AudioService.capture",
            )
            await loop.run_in_executor(
                self._capture_executor, self._open_capture_resources, None
            )

            self._playback_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="AudioService.playback",
            )
            await loop.run_in_executor(
                self._playback_executor, self._open_playback_resources, None
            )

        except BaseException:
            await self._stop()
            raise

    def _open_capture_resources(self, _: object = None) -> None:
        stack = ExitStack()

        try:
            self._input_stream.open()
            stack.callback(self._input_stream.close)

            self._vad_model.open()
            stack.callback(self._vad_model.close)

            self._wakeword_model.open()
            stack.callback(self._wakeword_model.close)
        except BaseException:
            stack.close()
            raise

        self._capture_resources = stack

    def _open_playback_resources(self, _: object = None) -> None:
        self._output_stream.open()

    async def _stop(self) -> None:
        try:
            await self._stop_capture()
        finally:
            await self._stop_playback()

    async def _stop_capture(self):
        executor = self._capture_executor
        if executor is None:
            return

        loop = asyncio.get_running_loop()
        self._input_cancel.set()

        try:
            if self._capture_future is not None:
                await self._capture_future
        finally:
            try:
                await loop.run_in_executor(
                    executor,
                    self._close_capture_resources,
                    None,
                )
            finally:
                try:
                    await asyncio.to_thread(executor.shutdown)
                finally:
                    self._capture_future = None
                    self._capture_executor = None
                    self._logger.debug("Capture executor STOPPED")

    async def _stop_playback(self) -> None:
        executor = self._playback_executor
        if executor is None:
            return

        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(
                executor,
                self._close_playback_resources,
                None,
            )
        finally:
            try:
                await asyncio.to_thread(executor.shutdown)
            finally:
                self._playback_executor = None
                self._logger.debug("Playback executor STOPPED")

    def _close_capture_resources(self, _: object = None) -> None:
        stack = self._capture_resources
        self._capture_resources = None

        if stack is not None:
            stack.close()

    def _close_playback_resources(self, _: object = None) -> None:
        self._output_stream.close()
