from types import SimpleNamespace

import pytest

from henry_client.reply import ReplyChunk, ReplySignal
from henry_client.reply.adapters import mlx_lm
from henry_client.reply.adapters.mlx_lm import MLXResponder, MLXResponderConfig


class FakeTokenizer:
    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []

    def apply_chat_template(self, *_args, **_kwargs) -> str:
        self.messages.append([message.copy() for message in _args[0]])
        return "prompt"


def test_mlx_responder_streams_model_output_as_reply_chunks(monkeypatch) -> None:
    monkeypatch.setattr(
        MLXResponder,
        "_load_model",
        lambda _self: (object(), FakeTokenizer()),
    )
    monkeypatch.setattr(
        MLXResponder,
        "_generate",
        lambda _self, _prompt: iter(
            [SimpleNamespace(text="First"), SimpleNamespace(text=" second")]
        ),
    )
    responder = MLXResponder(MLXResponderConfig(model_id="model"))

    responder.open()
    try:
        replies = list(responder.respond("Question"))
    finally:
        responder.close()

    assert replies == [ReplyChunk("First"), ReplyChunk(" second")]


def test_mlx_responder_replies_to_wakeword_with_configured_chunk(
    monkeypatch,
) -> None:
    delays: list[float] = []
    monkeypatch.setattr(mlx_lm.time, "sleep", delays.append)
    responder = MLXResponder(
        MLXResponderConfig(
            model_id="model",
            activation_text="Ready.",
            activation_start_delay=0.25,
        )
    )

    replies = list(responder.respond(ReplySignal.ACTIVATION))

    assert replies == [ReplyChunk("Ready.")]
    assert delays == [0.25]


def test_mlx_responder_rejects_negative_wakeword_delay() -> None:
    with pytest.raises(ValueError, match="delay cannot be negative"):
        MLXResponderConfig(
            model_id="model",
            activation_start_delay=-0.1,
        )


def test_mlx_responder_keeps_system_prompt_outside_bounded_history(
    monkeypatch,
) -> None:
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        MLXResponder,
        "_load_model",
        lambda _self: (object(), tokenizer),
    )
    monkeypatch.setattr(
        MLXResponder,
        "_generate",
        lambda _self, _prompt: iter([SimpleNamespace(text="Answer")]),
    )
    responder = MLXResponder(
        MLXResponderConfig(
            model_id="model",
            system_prompt="Always follow this instruction.",
        )
    )

    responder.open()
    try:
        for index in range(4):
            list(responder.respond(f"Question {index}"))
    finally:
        responder.close()

    assert len(tokenizer.messages) == 4
    assert all(
        messages[0]
        == {
            "role": "system",
            "content": "Always follow this instruction.",
        }
        for messages in tokenizer.messages
    )
    assert all(
        len(messages) <= responder._MAX_MESSAGES_LEN + 1
        for messages in tokenizer.messages
    )
