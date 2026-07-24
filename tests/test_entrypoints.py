import inspect

import pytest

from henry_cli import main as cli
from henry_cli.main import main as cli_main
from henry_debugger import main as debugger
from henry_debugger.main import main as debugger_main

ENV_VARS = (
    "HENRY_LOG_LEVEL",
    "HENRY_PROFILE_KIND",
    "HENRY_PROFILE_NAME",
    "HENRY_SYSTEM_LANGUAGE",
    "HENRY_WAKEWORD_REPLY",
    "HENRY_WAKEWORD_MODEL",
    "HENRY_VOICE_MODEL",
    "HENRY_LANGUAGE_MODEL",
    "HENRY_MAX_EMPTY_SEGMENTS",
)


def test_console_entrypoints_are_synchronous() -> None:
    assert not inspect.iscoroutinefunction(cli_main)
    assert not inspect.iscoroutinefunction(debugger_main)


@pytest.mark.parametrize(
    ("parse_args", "expected"),
    [
        (
            cli._parse_args,
            {
                "log_level": "DEBUG",
                "profile_kind": "default",
                "profile_name": "Henry",
                "system_language": "Polish",
                "wakeword_reply": "Tak, Wielmożny Panie...",
                "wakeword_model": "Hey_Henree_20260406_162745.onnx",
                "voice_model": "pl/pl_PL/bass/high/pl_PL-bass-high.onnx",
                "language_model": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
                "max_empty_segments": 3,
            },
        ),
        (
            debugger._parse_args,
            {
                "log_level": "DEBUG",
                "profile_kind": "sarcastic",
                "profile_name": "Alexa",
                "system_language": "Polish",
                "wakeword_reply": "Tak Słucham...",
                "wakeword_model": "alexa_v0.1.onnx",
                "voice_model": "pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx",
                "language_model": "mlx-community/Qwen3.5-4B-MLX-4bit",
                "max_empty_segments": 3,
            },
        ),
    ],
)
def test_arguments_preserve_main_defaults(monkeypatch, parse_args, expected) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    assert vars(parse_args([])) == expected


@pytest.mark.parametrize("parse_args", [cli._parse_args, debugger._parse_args])
def test_command_arguments_override_environment(monkeypatch, parse_args) -> None:
    monkeypatch.setenv("HENRY_PROFILE_NAME", "From environment")

    config = parse_args(["--profile-name", "From command"])

    assert config.profile_name == "From command"


@pytest.mark.parametrize("parse_args", [cli._parse_args, debugger._parse_args])
def test_environment_overrides_default(monkeypatch, parse_args) -> None:
    values = {
        "HENRY_LOG_LEVEL": "INFO",
        "HENRY_PROFILE_KIND": "friendly",
        "HENRY_PROFILE_NAME": "Ada",
        "HENRY_SYSTEM_LANGUAGE": "English",
        "HENRY_WAKEWORD_REPLY": "Ready.",
        "HENRY_WAKEWORD_MODEL": "wakeword.onnx",
        "HENRY_VOICE_MODEL": "voice.onnx",
        "HENRY_LANGUAGE_MODEL": "local/model",
        "HENRY_MAX_EMPTY_SEGMENTS": "5",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    config = parse_args([])

    assert vars(config) == {
        "log_level": "INFO",
        "profile_kind": "friendly",
        "profile_name": "Ada",
        "system_language": "English",
        "wakeword_reply": "Ready.",
        "wakeword_model": "wakeword.onnx",
        "voice_model": "voice.onnx",
        "language_model": "local/model",
        "max_empty_segments": 5,
    }
