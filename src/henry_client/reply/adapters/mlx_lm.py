from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from loguru import logger

from ..domain import ReplyChunk, ReplyRequest, ReplySignal
from ..ports import Responder

if TYPE_CHECKING:
    from mlx.nn import Module as MLXModel
    from mlx_lm.generate import GenerationResponse
    from mlx_lm.tokenizer_utils import TokenizerWrapper as MLXTokenizer


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    content: str
    role: MessageRole


@dataclass(frozen=True, slots=True)
class MLXResponderConfig:
    model_id: str
    max_tokens: int = 512
    system_prompt: str | None = None
    activation_text: str | None = None
    activation_start_delay: float = 0.5

    def __post_init__(self) -> None:
        if self.activation_start_delay < 0:
            raise ValueError("Activation delay cannot be negative")


class MLXResponderError(RuntimeError): ...


class MLXResponder(Responder):
    """Generate replies with MLX while retaining bounded conversation history."""

    _MAX_MESSAGES_LEN = 5

    def __init__(
        self,
        config: MLXResponderConfig,
    ) -> None:
        self._messages: list[Message] = list()
        self._config = config
        self._model: MLXModel | None = None
        self._tokenizer: MLXTokenizer | None = None
        self._logger = logger.bind(component="MLXResponder")

    def respond(self, request: ReplyRequest) -> Iterator[ReplyChunk]:
        if isinstance(request, ReplySignal):
            yield from self._respond_to_signal(request)
            return

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not loaded")

        self._append_message(
            Message(
                role=MessageRole.USER,
                content=request,
            )
        )

        messages = [
            {
                "role": msg.role.value,
                "content": msg.content,
            }
            for msg in self._messages
        ]
        if self._config.system_prompt is not None:
            # The system instruction is not part of the bounded conversation history.
            messages.insert(
                0,
                {
                    "role": MessageRole.SYSTEM.value,
                    "content": self._config.system_prompt,
                },
            )

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(prompt, str):
            raise MLXResponderError("Chat template did not produce text")

        content = ""

        for chunk in self._generate(prompt):
            content += chunk.text
            yield ReplyChunk(chunk.text)

        if content:
            self._append_message(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=content,
                )
            )

    def open(self) -> None:
        if self._model is not None:
            raise MLXResponderError("Model is already loaded")

        self._logger.debug("Loading model: model_id='{}'", self._config.model_id)
        self._model, self._tokenizer = self._load_model()

        self._logger.debug("Model READY")

    def close(self) -> None:
        if self._model is None:
            return

        self._messages.clear()
        self._model = None
        self._tokenizer = None
        self._logger.debug("Model CLOSED")

    def _respond_to_signal(self, signal: ReplySignal) -> Iterator[ReplyChunk]:
        match signal:
            case ReplySignal.ACTIVATION:
                if self._config.activation_text is None:
                    return

                time.sleep(self._config.activation_start_delay)
                yield ReplyChunk(self._config.activation_text)

    def _load_model(self) -> tuple[MLXModel, MLXTokenizer]:
        from huggingface_hub.utils import disable_progress_bars
        from mlx_lm import load

        with disable_progress_bars():
            loaded = load(
                self._config.model_id,
                return_config=False,
            )

        return loaded[0], loaded[1]

    def _generate(self, prompt: str) -> Iterator[GenerationResponse]:
        from mlx_lm import stream_generate

        assert self._model is not None
        assert self._tokenizer is not None
        return stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self._config.max_tokens,
        )

    def _append_message(self, message: Message) -> None:
        self._messages.append(message)
        del self._messages[: -self._MAX_MESSAGES_LEN]
