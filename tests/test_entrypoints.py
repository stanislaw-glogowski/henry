import asyncio
import inspect

from henry_cli import main as cli
from henry_cli.main import main as cli_main

ENV_VARS = (
    "HENRY_LOG_LEVEL",
    "HENRY_PROFILE_KIND",
    "HENRY_PROFILE_NAME",
    "HENRY_SYSTEM_LANGUAGE",
    "HENRY_WAKEWORD_REPLY",
    "HENRY_WAKEWORD_MODEL",
    "HENRY_VOICE_MODEL",
    "HENRY_LANGUAGE_MODEL",
)


def test_console_entrypoints_are_synchronous() -> None:
    assert not inspect.iscoroutinefunction(cli_main)


def test_arguments_preserve_main_defaults(monkeypatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    assert vars(cli._parse_args([])) == {
        "no_ui": False,
        "log_level": "DEBUG",
        "profile_kind": "default",
        "profile_name": "Henry",
        "system_language": "Polish",
        "wakeword_reply": "Tak, Wielmożny Panie...",
        "wakeword_model": "Hey_Henree_20260406_162745.onnx",
        "voice_model": "pl/pl_PL/bass/high/pl_PL-bass-high.onnx",
        "language_model": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
    }


def test_command_arguments_override_environment(monkeypatch) -> None:
    monkeypatch.setenv("HENRY_PROFILE_NAME", "From environment")

    config = cli._parse_args(["--profile-name", "From command"])

    assert config.profile_name == "From command"


def test_environment_overrides_default(monkeypatch) -> None:
    values = {
        "HENRY_LOG_LEVEL": "INFO",
        "HENRY_PROFILE_KIND": "friendly",
        "HENRY_PROFILE_NAME": "Ada",
        "HENRY_SYSTEM_LANGUAGE": "English",
        "HENRY_WAKEWORD_REPLY": "Ready.",
        "HENRY_WAKEWORD_MODEL": "wakeword.onnx",
        "HENRY_VOICE_MODEL": "voice.onnx",
        "HENRY_LANGUAGE_MODEL": "local/model",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    config = cli._parse_args([])

    assert vars(config) == {
        "no_ui": False,
        "log_level": "INFO",
        "profile_kind": "friendly",
        "profile_name": "Ada",
        "system_language": "English",
        "wakeword_reply": "Ready.",
        "wakeword_model": "wakeword.onnx",
        "voice_model": "voice.onnx",
        "language_model": "local/model",
    }


def test_no_ui_argument_supports_requested_and_long_forms() -> None:
    assert cli._parse_args(["-noui"]).no_ui
    assert cli._parse_args(["--no-ui"]).no_ui


def test_no_ui_runs_with_console_event_logging(monkeypatch) -> None:
    recorded = {}

    class FakeApp:
        def __init__(self, config, events) -> None:
            recorded["config"] = config
            recorded["events"] = events

        async def run(self, shutdown) -> None:
            recorded["shutdown"] = shutdown

    def fail_layout(**_):
        raise AssertionError("UI must not be constructed in no-UI mode")

    monkeypatch.setattr(cli, "App", FakeApp)
    monkeypatch.setattr(cli, "Layout", fail_layout)
    monkeypatch.setattr(cli, "_configure_shutdown", asyncio.Event)
    monkeypatch.setattr(
        cli,
        "configure_console_logger",
        lambda level: recorded.update(log_level=level),
    )

    asyncio.run(cli.run(cli._parse_args(["-noui", "--log-level", "TRACE"])))

    assert isinstance(recorded["events"], cli.EventLogger)
    assert recorded["log_level"] == "TRACE"
