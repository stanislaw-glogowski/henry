import asyncio
import queue
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from loguru import logger
from piper import AudioChunk, PiperVoice

from ...concurrency import (
    set_future_exception_if_pending,
    set_future_result_if_pending,
)
from ...domain import AudioFormat, AudioFrame
from ...ports import SpeechSynthesizer

WORKER_NAME = "piper-synthesizer"
REPO_ID = "rhasspy/piper-voices"


type Response = AudioChunk | Failure | None
type RequestQueue = queue.Queue[Request | None]


@dataclass(frozen=True, slots=True)
class Request:
    text: str
    responses: asyncio.Queue[Response]


@dataclass(frozen=True, slots=True)
class Failure:
    error: BaseException


class PiperSynthesizer(SpeechSynthesizer):
    def __init__(self, voice_path: str) -> None:
        self._voice_path = voice_path
        self._ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._requests: RequestQueue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._logger = logger.bind(component="PiperSynthesizer")

    async def __aenter__(self) -> Self:
        loop = asyncio.get_running_loop()
        self._worker = threading.Thread(
            target=self._run_worker,
            args=(loop,),
            name=WORKER_NAME,
        )

        assert self._worker is not None
        self._worker.start()

        self._logger.trace("Worker STARTED")

        await self._ready
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._requests.put_nowait(None)
        assert self._worker is not None
        await asyncio.to_thread(
            self._worker.join,
        )

        self._logger.trace("Worker STOPPED")

        return None

    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        request = Request(
            text=text,
            responses=asyncio.Queue(),
        )

        self._requests.put_nowait(request)

        audio_format: AudioFormat | None = None

        while True:
            chunk = await request.responses.get()

            if chunk is None:
                return

            if isinstance(chunk, Failure):
                raise chunk.error

            assert isinstance(chunk, AudioChunk)

            if audio_format is None:
                audio_format = AudioFormat(
                    sample_rate=chunk.sample_rate,
                    channels=chunk.sample_channels,
                )

            assert audio_format is not None

            yield AudioFrame(
                samples=np.ascontiguousarray(
                    chunk.audio_float_array,
                    dtype=np.float32,
                ),
                format=audio_format,
            )

    def _require_worker(self) -> None:
        if self._worker is None:
            raise RuntimeError("Worker not initialized")

    def _run_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        try:

            with disable_progress_bars():
                self._logger.trace("Loading model: voice_path='{}'", self._voice_path)

                model_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=self._voice_path,
                )
                config_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=self._voice_path + ".json",
                )

                model = PiperVoice.load(model_path=model_path, config_path=config_path)
        except BaseException as error:
            loop.call_soon_threadsafe(
                set_future_exception_if_pending,
                self._ready,
                error,
            )
            return

        loop.call_soon_threadsafe(
            set_future_result_if_pending,
            self._ready,
            None,
        )

        self._logger.trace("Worker READY")

        while True:
            request = self._requests.get()
            try:
                if request is None:
                    return

                self._logger.trace("Request RECEIVED")

                chunks = model.synthesize(request.text)

                for chunk in chunks:
                    loop.call_soon_threadsafe(
                        request.responses.put_nowait,
                        chunk,
                    )

                loop.call_soon_threadsafe(
                    request.responses.put_nowait,
                    None,
                )

            except BaseException as error:
                if request is not None:
                    loop.call_soon_threadsafe(
                        request.responses.put_nowait,
                        Failure(error=error),
                    )
            finally:
                self._requests.task_done()
                if request is not None:
                    self._logger.trace("Request PROCESSED")
