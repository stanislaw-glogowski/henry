import asyncio
import inspect
import runpy

import pytest

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
    "HENRY_MAX_EMPTY_SEGMENTS",
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
        "max_empty_segments": 3,
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
        "HENRY_MAX_EMPTY_SEGMENTS": "5",
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
        "max_empty_segments": 5,
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


def test_ui_mode_runs_app_and_ui_tasks(monkeypatch) -> None:
    recorded = []

    class FakeApp:
        def __init__(self, config, events) -> None:
            recorded.append(("app_config", config))

        async def run(self, shutdown) -> None:
            recorded.append("app")

    class FakeLayout:
        def __init__(self, logs, events) -> None:
            recorded.append("layout")

    async def run_ui(layout, shutdown) -> None:
        recorded.append("ui")

    async def stop_ui(layout, shutdown) -> None:
        recorded.append("stop")

    monkeypatch.setattr(cli, "App", FakeApp)
    monkeypatch.setattr(cli, "Layout", FakeLayout)
    monkeypatch.setattr(cli, "LogBuffer", lambda level: object())
    monkeypatch.setattr(cli, "_configure_shutdown", asyncio.Event)
    monkeypatch.setattr(cli, "_run_ui", run_ui)
    monkeypatch.setattr(cli, "_stop_ui_on_shutdown", stop_ui)

    asyncio.run(cli.run(cli._parse_args([])))

    assert "layout" in recorded
    assert set(recorded[2:]) == {"app", "ui", "stop"}


def test_invalid_environment_profile_and_empty_segment_limit_are_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HENRY_PROFILE_KIND", "invalid")
    with pytest.raises(SystemExit):
        cli._parse_args([])

    monkeypatch.setenv("HENRY_PROFILE_KIND", "default")
    with pytest.raises(SystemExit):
        cli._parse_args(["--max-empty-segments", "0"])


def test_main_parses_arguments_and_runs_async_entrypoint(monkeypatch) -> None:
    recorded = {}
    config = object()

    async def fake_run(value) -> None:
        recorded["config"] = value

    monkeypatch.setattr(cli, "_parse_args", lambda argv: config)
    monkeypatch.setattr(cli, "run", fake_run)

    cli.main(["--no-ui"])

    assert recorded["config"] is config


def test_configure_shutdown_registers_both_signals(monkeypatch) -> None:
    handlers = {}

    class FakeLoop:
        def add_signal_handler(self, signal_number, callback, argument) -> None:
            handlers[signal_number] = (callback, argument)

    monkeypatch.setattr(cli.asyncio, "get_running_loop", lambda: FakeLoop())

    shutdown = cli._configure_shutdown()
    callback, argument = handlers[cli.signal.SIGINT]
    callback(argument)

    assert set(handlers) == {cli.signal.SIGINT, cli.signal.SIGTERM}
    assert shutdown.is_set()


def test_ui_helpers_propagate_failure_and_stop_layout() -> None:
    class FakeLayout:
        def __init__(self) -> None:
            self.exited = False

        async def run_async(self) -> None:
            raise RuntimeError("UI failed")

        def exit(self) -> None:
            self.exited = True

    async def scenario() -> None:
        layout = FakeLayout()
        shutdown = asyncio.Event()

        with pytest.raises(RuntimeError, match="UI failed"):
            await cli._run_ui(layout, shutdown)
        assert shutdown.is_set()

        await cli._stop_ui_on_shutdown(layout, shutdown)
        assert layout.exited

    asyncio.run(scenario())


def test_module_entrypoint_calls_main(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "main", lambda: calls.append(True))

    runpy.run_module("henry_cli_old.__main__", run_name="__main__")

    assert calls == [True]
