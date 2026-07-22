import asyncio
import signal
import sys
from collections.abc import Callable

import loguru
from loguru import logger

from henry_client import (
    App,
    AppConfig,
    AppEvent,
    AppEventSink,
    Profile,
    ProfileKind,
)

LOG_LEVEL = "DEBUG"

ALEXA_PROFILE = Profile.build(
    kind=ProfileKind.SARCASTIC,
    name="Alexa",
    system_language="Polish",
    wakeword_model="alexa_v0.1.onnx",
    wakeword_reply="Tak Słucham...",
    voice_model="pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx",
)


APP_CONFIG = AppConfig(
    profile=ALEXA_PROFILE,
    language_model="mlx-community/Qwen3.5-4B-MLX-4bit",
)


class EventLogger(AppEventSink):
    def __init__(self) -> None:
        self._logger = logger.bind(component="EventTracker")

    def publish(self, *events: AppEvent) -> None:
        for event in events:
            self._logger.trace("{}", event, event=event.__class__.__name__)


def configure_logger(
    record_filter: Callable[[str, str], bool] | None = None,
) -> loguru.Logger:
    def _filter(record: loguru.Record) -> bool:
        if record_filter is None:
            return True

        component = record["extra"].get("component")
        event = record["extra"].get("event")

        assert isinstance(component, str)
        assert isinstance(event, str)

        return record_filter(component, event)

    logger.remove()
    logger.configure(
        extra={"component": "App", "event": ""},
    )
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <6}</level> | "
            "<magenta>@{extra[component]}</magenta> | "
            "<level>{message}</level>"
        ),
        filter=_filter,
        colorize=True,
    )

    return logger


def configure_shutdown() -> asyncio.Event:
    shutdown = asyncio.Event()

    def signal_handler(_: object) -> None:
        shutdown.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler, None)
    loop.add_signal_handler(signal.SIGTERM, signal_handler, None)
    return shutdown


async def run() -> None:
    """Run the assistant with event logging instead of the terminal UI."""
    configure_logger()

    await App(config=APP_CONFIG, events=EventLogger()).run(configure_shutdown())


def main() -> None:
    """Console-script entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
