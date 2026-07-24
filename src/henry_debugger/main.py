import argparse
import asyncio
import os
import signal
import sys
from collections.abc import Callable, Sequence

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


class EventLogger(AppEventSink):
    def __init__(self) -> None:
        self._logger = logger.bind(component="EventTracker")

    def publish(self, *events: AppEvent) -> None:
        for event in events:
            self._logger.trace("{}", event, event=event.__class__.__name__)


def configure_logger(
    record_filter: Callable[[str, str], bool] | None = None,
    level: str = "DEBUG",
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
        level=level,
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


async def run(config: argparse.Namespace) -> None:
    """Run the assistant with event logging instead of the terminal UI."""
    configure_logger(level=config.log_level)

    await App(
        config=AppConfig(
            profile=Profile.build(
                kind=ProfileKind[config.profile_kind.upper()],
                name=config.profile_name,
                system_language=config.system_language,
                wakeword_model=config.wakeword_model,
                wakeword_reply=config.wakeword_reply,
                voice_model=config.voice_model,
            ),
            language_model=config.language_model,
            max_empty_segments=config.max_empty_segments,
        ),
        events=EventLogger(),
    ).run(configure_shutdown())


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point."""
    asyncio.run(run(_parse_args(argv)))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Henry with event logging.")
    profile_kinds = [kind.name.lower() for kind in ProfileKind]
    parser.add_argument(
        "--log-level",
        default=os.getenv("HENRY_LOG_LEVEL", "DEBUG"),
        help="Log level (env: HENRY_LOG_LEVEL).",
    )
    parser.add_argument(
        "--profile-kind",
        choices=profile_kinds,
        default=os.getenv("HENRY_PROFILE_KIND", "sarcastic").lower(),
        help="Assistant profile kind (env: HENRY_PROFILE_KIND).",
    )
    parser.add_argument(
        "--profile-name",
        default=os.getenv("HENRY_PROFILE_NAME", "Alexa"),
        help="Assistant name (env: HENRY_PROFILE_NAME).",
    )
    parser.add_argument(
        "--system-language",
        default=os.getenv("HENRY_SYSTEM_LANGUAGE", "Polish"),
        help="Language used in the system prompt (env: HENRY_SYSTEM_LANGUAGE).",
    )
    parser.add_argument(
        "--wakeword-reply",
        default=os.getenv("HENRY_WAKEWORD_REPLY", "Tak Słucham..."),
        help="Reply after wake-word detection (env: HENRY_WAKEWORD_REPLY).",
    )
    parser.add_argument(
        "--wakeword-model",
        default=os.getenv("HENRY_WAKEWORD_MODEL", "alexa_v0.1.onnx"),
        help="Wake-word model path (env: HENRY_WAKEWORD_MODEL).",
    )
    parser.add_argument(
        "--voice-model",
        default=os.getenv(
            "HENRY_VOICE_MODEL", "pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx"
        ),
        help="Piper voice model path (env: HENRY_VOICE_MODEL).",
    )
    parser.add_argument(
        "--language-model",
        default=os.getenv("HENRY_LANGUAGE_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit"),
        help="MLX language model id or path (env: HENRY_LANGUAGE_MODEL).",
    )
    parser.add_argument(
        "--max-empty-segments",
        type=int,
        default=os.getenv("HENRY_MAX_EMPTY_SEGMENTS", "3"),
        help=(
            "Empty utterance timeouts before returning to wake-word mode "
            "(env: HENRY_MAX_EMPTY_SEGMENTS)."
        ),
    )
    config = parser.parse_args(argv)
    if config.profile_kind not in profile_kinds:
        parser.error("HENRY_PROFILE_KIND must be one of: " + ", ".join(profile_kinds))
    if config.max_empty_segments < 1:
        parser.error("HENRY_MAX_EMPTY_SEGMENTS must be positive")
    return config


if __name__ == "__main__":
    main()
