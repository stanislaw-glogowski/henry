from collections.abc import Iterator, Sequence
from types import TracebackType
from typing import Self

from huggingface_hub.utils import disable_progress_bars
from loguru import logger
from mlx.nn import Module as Model
from mlx_lm import load, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper as Tokenizer

from ..domain import Message, MessageChunk
from ..ports import LanguageModel

MAX_TOKENS = 512


class MLXLanguageModelError(RuntimeError): ...


class MLXLanguageModel(LanguageModel):
    def __init__(
        self,
        model_id: str,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._model: Model | None = None
        self._tokenizer: Tokenizer | None = None
        self._logger = logger.bind(component="MLXLanguageModel")

    def __enter__(self) -> Self:
        self._open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close()

    def generate(self, messages: Sequence[Message]) -> Iterator[MessageChunk]:
        model, tokenizer = self._require_model()

        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": message.role.value,
                    "content": message.content,
                }
                for message in messages
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        stream = stream_generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self._max_tokens,
        )

        for chunk in stream:
            text = chunk.text
            yield MessageChunk(
                content=text,
            )

    def _open(self) -> None:
        if self._model is not None:
            raise MLXLanguageModelError("Model is already loaded")

        with disable_progress_bars():
            self._logger.debug("Loading model: model_id='{}'", self._model_id)
            self._model, self._tokenizer = load(self._model_id)

        self._logger.debug("Model READY")

    def _close(self) -> None:
        if self._model is None:
            return

        self._model = None
        self._tokenizer = None
        self._logger.debug("Model CLOSED")

    def _require_model(self) -> tuple[Model, Tokenizer]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not loaded")

        return self._model, self._tokenizer
