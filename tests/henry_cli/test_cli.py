import asyncio
import runpy
import signal
from types import SimpleNamespace

import pytest
from loguru import logger

import henry_cli.main as main_module
from henry_cli.events import run_event_logger
from henry_cli.logger import _ensure_component, configure_console_logger
from henry_common.events import EventBus, ShutdownEvent
from henry_conversation import (
    ConversationActivated,
    GenerateReply,
    ReplyCompleted,
    ReplyLine,
    ReplyStarted,
    UserTurn,
)
from henry_conversation.config import ConversationProfile, ConversationPrompts
from henry_speech.config import SpeechSettings
from henry_speech.events import WakeWordObserved


def profile():
    return SimpleNamespace(
        name="Henry",
        language="Polish",
        conversation=ConversationProfile(
            model="test:model",
            prompts=ConversationPrompts(system="s", opening="o", summary="m"),
        ),
    )


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
                ReplyStarted(),
                ReplyLine("Answer"),
                ReplyCompleted(),
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


def test_run_uses_default_profile_and_starts_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        calls: list[tuple] = []
        selected: list[str] = []
        fake_profile = profile()
        store = SimpleNamespace(
            load_profile=lambda name: selected.append(name) or fake_profile,
            load_settings=lambda: SimpleNamespace(speech=SpeechSettings()),
        )

        async def conversation(bus, context) -> None:
            calls.append(("conversation", bus, context))

        async def speech(profile_value, settings, store_value, bus) -> None:
            calls.append(("speech", profile_value, settings, store_value, bus))

        async def events(bus) -> None:
            calls.append(("events", bus))

        monkeypatch.setattr(main_module, "LocalStore", lambda: store)
        monkeypatch.setattr(main_module, "configure_console_logger", lambda: None)
        monkeypatch.setattr(main_module, "configure_shutdown", lambda _: None)
        monkeypatch.setattr(main_module, "run_conversation_worker", conversation)
        monkeypatch.setattr(main_module, "run_speech_worker", speech)
        monkeypatch.setattr(main_module, "run_event_logger", events)

        await main_module.run()

        assert selected == ["default"]
        assert {call[0] for call in calls} == {"conversation", "speech", "events"}
        conversation_call = next(call for call in calls if call[0] == "conversation")
        assert conversation_call[2].model == "test:model"
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
