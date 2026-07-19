import asyncio
import queue
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from huggingface_hub.utils import disable_progress_bars
from loguru import logger
from mlx_lm import generate, load

from ...concurrency import (
    set_future_exception_if_pending,
    set_future_result_if_pending,
)
from ...domain import ConversationMessage
from ...ports import LLModel

WORKER_NAME = "mlx-llmodel"


type RequestQueue = queue.Queue[Request | None]


@dataclass(frozen=True, slots=True)
class Request:
    messages: tuple[ConversationMessage, ...]
    max_tokens: int
    response: asyncio.Future[str]


class MlxLLModel(LLModel):
    def __init__(self, model_id: str, max_tokens: int) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._requests: RequestQueue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._logger = logger.bind(component="MlxLLModel")

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

    async def generate_reply(
        self,
        messages: Sequence[ConversationMessage],
    ) -> str:
        self._require_worker()

        request = Request(
            messages=tuple(messages),
            response=asyncio.get_running_loop().create_future(),
            max_tokens=self._max_tokens,
        )

        self._requests.put_nowait(request)

        return await request.response

    def _require_worker(self) -> None:
        if self._worker is None:
            raise RuntimeError("Worker not initialized")

    def _run_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            with disable_progress_bars():
                self._logger.trace("Loading model: model_id='{}'", self._model_id)
                model, tokenizer = load(self._model_id)

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

                messages = [
                    {
                        "role": message.role.value,
                        "content": message.content,
                    }
                    for message in request.messages
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                text = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    verbose=False,
                ).strip()

                loop.call_soon_threadsafe(
                    set_future_result_if_pending,
                    request.response,
                    text,
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
