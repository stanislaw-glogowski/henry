import asyncio
import queue
import threading
from dataclasses import dataclass
from types import TracebackType
from typing import Self, cast

import mlx.core as mx
import numpy as np
from domain import SpeechSegment, SpeechTranscription
from huggingface_hub.utils import disable_progress_bars
from loguru import logger
from mlx_audio.stt.models.nemo.alignment import AlignedResult
from mlx_audio.stt.models.parakeet import Model
from mlx_audio.stt.utils import load_model

from ...concurrency import (
    set_future_exception_if_pending,
    set_future_result_if_pending,
)
from ...ports import SpeechTranscriber

MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
WORKER_NAME = "parakeet-transcriber"


type Response = AlignedResult | None
type RequestQueue = queue.Queue[Request | None]


@dataclass(frozen=True, slots=True)
class Request:
    audio: np.ndarray
    response: asyncio.Future[Response]


class ParakeetTranscriber(SpeechTranscriber):
    def __init__(self) -> None:
        self._ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._requests: RequestQueue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._logger = logger.bind(component="ParakeetTranscriber")

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

    async def transcribe(self, segment: SpeechSegment) -> SpeechTranscription | None:
        self._require_worker()

        request = Request(
            audio=segment.audio.samples,
            response=asyncio.get_running_loop().create_future(),
        )

        self._requests.put_nowait(request)

        response = await request.response

        if response is None:
            return None

        text = response.text.strip()

        self._logger.trace("Transcription: text='{}'", text)

        if text == "":
            return None

        return SpeechTranscription(
            text=text,
        )

    def _require_worker(self) -> None:
        if self._worker is None:
            raise RuntimeError("Worker not initialized")

    def _run_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            with disable_progress_bars():
                self._logger.trace("Loading model: model_id='{}'", MODEL_ID)
                model = cast(Model, load_model(MODEL_ID))

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

                raw_result = model.generate(mx.array(request.audio))
                result: AlignedResult | None = None

                if isinstance(raw_result, AlignedResult):
                    result = raw_result

                loop.call_soon_threadsafe(
                    set_future_result_if_pending,
                    request.response,
                    result,
                )
            except BaseException as error:
                if request is not None:
                    loop.call_soon_threadsafe(
                        set_future_exception_if_pending,
                        request.response,
                        error,
                    )
            finally:
                self._requests.task_done()
                if request is not None:
                    self._logger.trace("Request PROCESSED")
