import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import ExitStack

from henry_common.components import AbstractAsyncService

from ..audio import AudioInput
from .domain import DetectionResult, SpeechChunk
from .ports import VADModel, WakeWordModel


class CaptureService(AbstractAsyncService):
    def __init__(
        self,
        audio_input: AudioInput,
        vad_model: VADModel,
        wakeword_model: WakeWordModel,
    ):
        super().__init__()
        self._audio_input = audio_input
        self._vad_model = vad_model
        self._wakeword_enabled = threading.Event()
        self._wakeword_reset = threading.Event()
        self._wakeword_model = wakeword_model
        self._capture_cancel = threading.Event()
        self._capture_future: asyncio.Future[None] | None = None
        self._resources: ExitStack | None = None

    def enable_wakeword(self) -> None:
        """Arm wake-word detection."""
        if self._wakeword_enabled.is_set():
            return

        self._wakeword_reset.set()
        self._wakeword_enabled.set()

    def disable_wakeword(self) -> None:
        if not self._wakeword_enabled.is_set():
            return

        """Disarm wake-word detection."""
        self._wakeword_enabled.clear()

    async def capture(self) -> AsyncIterator[SpeechChunk]:
        if self._capture_future is not None:
            raise RuntimeError("Capture is already running")

        loop = asyncio.get_running_loop()
        responses = asyncio.Queue()

        self._capture_future = self._run_in_executor(
            self._run_capture_loop, loop, responses
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
            await self._cancel_capture_loop()

    def _run_capture_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        responses: asyncio.Queue[SpeechChunk | BaseException | None],
    ) -> None:
        try:
            while not self._capture_cancel.is_set():
                audio = self._audio_input.read()

                if self._wakeword_reset.is_set():
                    self._wakeword_model.reset()
                    self._wakeword_reset.clear()

                vad = self._vad_model.detect(audio)
                wakeword: DetectionResult | None = None

                if self._wakeword_enabled.is_set():
                    wakeword = self._wakeword_model.detect(audio)

                loop.call_soon_threadsafe(
                    responses.put_nowait,
                    SpeechChunk(
                        audio=audio,
                        vad=vad,
                        wakeword=wakeword,
                    ),
                )
            responses.put_nowait(None)
        except BaseException as err:
            responses.put_nowait(err)

    async def _cancel_capture_loop(self) -> None:
        if self._capture_future is None:
            return

        capture_future, self._capture_future = self._capture_future, None

        try:
            self._capture_cancel.set()
            await capture_future
        finally:
            self._capture_cancel.clear()
            self._capture_future = None

    async def _post_stop(self) -> None:
        await self._cancel_capture_loop()

    def _open_resources(self) -> None:
        if self._resources is not None:
            return

        stack = ExitStack()

        try:
            self._audio_input.open()
            stack.callback(self._audio_input.close)

            self._vad_model.open()
            stack.callback(self._vad_model.close)

            self._wakeword_enabled.clear()
            self._wakeword_reset.clear()
            self._wakeword_model.open()
            stack.callback(self._wakeword_model.close)
        except BaseException:
            stack.close()
            raise

        self._resources = stack

    def _close_resources(self) -> None:
        stack, self._resources = self._resources, None
        if stack is not None:
            stack.close()
