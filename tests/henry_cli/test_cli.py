import asyncio
import importlib
import os
import runpy
import signal
from types import SimpleNamespace

import pytest
from loguru import logger

import henry_cli
import henry_cli.main as main_module
from henry_cli.events import run_event_logger
from henry_cli.logger import _ensure_component, configure_console_logger
from henry_common.events import EventBus, ShutdownEvent
from henry_conversation import (
    ConversationActivated,
    GenerateReply,
    ReplyGenerationCompleted,
    ReplyGenerationStarted,
    ReplyPhrase,
    UserTurn,
)
from henry_conversation.config import ConversationSettings
from henry_conversation.profile import ConversationProfile, ConversationPrompts
from henry_resources import ProfileEntry, Settings
from henry_speech.config import SpeechSettings
from henry_speech.events import SpeechReady, WakeWordObserved

from .test_state import profile as runtime_profile


def profile():
    return SimpleNamespace(
        name="Henry",
        conversation=ConversationProfile(
            models={
                "fast": {"model_id": "test:model"},
                "detailed": {"model_id": "test:model"},
            },
            prompts=ConversationPrompts(system="s", opening="o", summary="m"),
        ),
    )


def test_cli_suppresses_unused_pytorch_advisory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", raising=False)

    importlib.reload(henry_cli)

    assert os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] == "1"


def test_console_logger_adds_default_component() -> None:
    record = {"extra": {}}
    assert _ensure_component(record)
    assert record["extra"] == {"component": "Henry"}

    configure_console_logger()
    messages: list[str] = []
    handler = logger.add(messages.append, format="{extra[component]}:{message}")
    try:
        logger.info("ready")
        logger.bind(component="Worker").debug("working")
    finally:
        logger.remove(handler)
    assert any("Henry:ready" in message for message in messages)
    assert any("Worker:working" in message for message in messages)


def test_event_logger_reports_conversation_events() -> None:
    async def scenario() -> None:
        bus = EventBus()
        messages: list[str] = []
        handler = logger.add(messages.append, format="{message}")
        try:
            task = asyncio.create_task(run_event_logger(bus))
            await asyncio.sleep(0)
            bus.publish(
                WakeWordObserved(score=0.9, detected=False),
                WakeWordObserved(score=0.9, detected=True),
                GenerateReply(ConversationActivated()),
                GenerateReply(UserTurn("Question")),
                ReplyGenerationStarted(1),
                ReplyPhrase(1, 1, "Answer"),
                ReplyGenerationCompleted(1),
                ShutdownEvent(),
            )
            await asyncio.wait_for(task, 1)
        finally:
            logger.remove(handler)

        output = "\n".join(messages)
        assert "Wake word detected" in output
        assert "Wake word activated the conversation" in output
        assert "User: Question" in output
        assert "Henry: Answer" in output
        assert "Response completed" in output
        assert "Shutdown requested" in output

    asyncio.run(scenario())


def test_configure_shutdown_registers_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    loop = SimpleNamespace(add_signal_handler=lambda *args: calls.append(args))
    monkeypatch.setattr(main_module.asyncio, "get_running_loop", lambda: loop)
    bus = EventBus()

    main_module.configure_shutdown(bus)

    assert [item[0] for item in calls] == [signal.SIGINT, signal.SIGTERM]
    assert all(item[1] == bus.publish for item in calls)
    assert all(isinstance(item[2], ShutdownEvent) for item in calls)


def test_profile_compatibility_validation_marks_only_bad_entries() -> None:
    valid_profile = runtime_profile()
    invalid = ProfileEntry("invalid", "Invalid", error="broken")
    incompatible_profile = runtime_profile(
        conversation={
            "models": {
                "fast": {"langchain": "test:fast"},
                "detailed": {"langchain": "test:detailed"},
            },
            "prompts": {
                "system": "system {conversation_summary}",
                "opening": "opening {conversation_summary}",
                "summary": "summary {conversation_summary} {recent_conversation}",
            },
        }
    )
    validated = main_module._validate_profiles(
        (
            ProfileEntry("valid", "Valid", valid_profile),
            invalid,
            ProfileEntry("incompatible", "Incompatible", incompatible_profile),
        ),
        Settings.model_validate(
            {"conversation": {"language_model": {"adapter": "mlx"}}}
        ),
    )
    assert validated[0].profile is valid_profile
    assert validated[1] is invalid
    assert validated[2].profile is None
    assert "Incompatible" in validated[2].error


def test_runtime_readiness_and_early_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        bus = EventBus()
        with bus.subscribe(
            main_module.ConversationReady,
            SpeechReady,
            ShutdownEvent,
        ) as events:
            ready = asyncio.create_task(main_module._wait_until_ready(events))
            bus.publish(main_module.ConversationReady(), SpeechReady())
            await asyncio.wait_for(ready, 1)

        with bus.subscribe(ShutdownEvent) as events:
            stopped = asyncio.create_task(main_module._wait_until_ready(events))
            bus.publish(ShutdownEvent())
            with pytest.raises(asyncio.CancelledError):
                await stopped

        async def successful(bus, *_args) -> None:
            start_event = _args[-1]
            bus.publish(main_module.ConversationReady(), SpeechReady())
            await start_event.wait()

        monkeypatch.setattr(main_module, "_run_workers", successful)
        backend = await main_module._start_runtime(
            bus,
            runtime_profile(),
            Settings(),
            object(),
        )
        await backend

        async def failing(*_args) -> None:
            raise RuntimeError("models failed")

        monkeypatch.setattr(main_module, "_run_workers", failing)
        with pytest.raises(RuntimeError, match="models failed"):
            await main_module._start_runtime(
                bus,
                runtime_profile(),
                Settings(),
                object(),
            )

    asyncio.run(scenario())


def test_run_selects_profile_then_starts_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        calls: list[tuple] = []
        fake_profile = profile()
        fake_settings = SimpleNamespace(
            conversation=ConversationSettings(), speech=SpeechSettings()
        )
        store = SimpleNamespace(
            inspect_profiles=lambda: [SimpleNamespace(profile=fake_profile)],
            load_settings=lambda: fake_settings,
        )

        class Bridge:
            def __init__(self) -> None:
                self.ready = asyncio.Event()

            async def wait_ready(self) -> None:
                await self.ready.wait()

            async def run(self, bus) -> None:
                with bus.subscribe(ShutdownEvent) as events:
                    self.ready.set()
                    event = await events.__anext__()
                    events.task_done()
                    assert isinstance(event, ShutdownEvent)

        class App:
            def __init__(self, *_args) -> None:
                self.exited = asyncio.Event()

            async def run_async(self) -> None:
                await self.exited.wait()

            async def wait_mounted(self) -> None:
                pass

            async def select_profile(self):
                return fake_profile

            def configure_runtime(self, profile_value, settings_value) -> None:
                calls.append(("runtime", profile_value, settings_value))

            async def show_startup(self) -> None:
                calls.append(("startup",))

            async def finish_startup(self) -> None:
                calls.append(("ready",))

            async def wait_quit_requested(self) -> None:
                pass

            def exit(self) -> None:
                self.exited.set()

        async def conversation(
            bus,
            conversation_profile,
            settings,
            start_event,
        ) -> None:
            calls.append(("conversation", conversation_profile, settings))
            with bus.subscribe(ShutdownEvent) as events:
                bus.publish(main_module.ConversationReady())
                await start_event.wait()
                await events.__anext__()
                events.task_done()

        async def speech(
            profile_value,
            settings,
            store_value,
            bus,
            start_event,
        ) -> None:
            calls.append(("speech", profile_value, settings, store_value))
            with bus.subscribe(ShutdownEvent) as events:
                bus.publish(main_module.SpeechReady())
                await start_event.wait()
                await events.__anext__()
                events.task_done()

        async def events(_bus) -> None:
            calls.append(("events",))

        monkeypatch.setattr(main_module, "LocalStore", lambda: store)
        monkeypatch.setattr(main_module, "UiEventBridge", Bridge)
        monkeypatch.setattr(main_module, "TerminalApp", App)
        monkeypatch.setattr(main_module, "_validate_profiles", lambda value, _: value)
        monkeypatch.setattr(main_module, "configure_ui_logger", lambda *_: None)
        monkeypatch.setattr(main_module, "configure_shutdown", lambda _: None)
        monkeypatch.setattr(main_module, "run_conversation_worker", conversation)
        monkeypatch.setattr(main_module, "run_speech_worker", speech)
        monkeypatch.setattr(main_module, "run_event_logger", events)

        await main_module.run()

        assert {call[0] for call in calls} == {
            "conversation",
            "events",
            "ready",
            "runtime",
            "speech",
            "startup",
        }
        conversation_call = next(call for call in calls if call[0] == "conversation")
        assert conversation_call[1] is fake_profile.conversation
        assert conversation_call[2] == ConversationSettings()
        speech_call = next(call for call in calls if call[0] == "speech")
        assert speech_call[1] is fake_profile
        assert speech_call[3] is store

    asyncio.run(scenario())


def test_main_runs_async_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[object] = []

    def fake_run(coroutine) -> None:
        captured.append(coroutine)
        coroutine.close()

    monkeypatch.setattr(main_module.asyncio, "run", fake_run)
    main_module.main()
    assert len(captured) == 1


def test_module_entrypoint_calls_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(main_module, "main", lambda: calls.append(True))
    runpy.run_module("henry_cli.__main__", run_name="__main__")
    assert calls == [True]
