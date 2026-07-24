import argparse
import asyncio
import os
import signal
from collections.abc import Sequence

from henry_cli.events import EventBridge, EventLogger
from henry_cli.logs import LogBuffer, configure_console_logger
from henry_cli.ui import Layout
from henry_client import App, AppConfig, Profile, ProfileKind


async def run(config: argparse.Namespace) -> None:
    """Run the assistant with terminal UI or console event logging."""
    shutdown = _configure_shutdown()

    app_config = AppConfig(
        profile=Profile.build(
            kind=ProfileKind[config.profile_kind.upper()],
            name=config.profile_name,
            system_language=config.system_language,
            wakeword_reply=config.wakeword_reply,
            wakeword_model=config.wakeword_model,
            voice_model=config.voice_model,
        ),
        language_model=config.language_model,
    )

    if config.no_ui:
        configure_console_logger(config.log_level)
        await App(config=app_config, events=EventLogger()).run(shutdown)
        return

    logs = LogBuffer(config.log_level)
    events = EventBridge()
    app = App(config=app_config, events=events)
    layout = Layout(
        logs=logs,
        events=events,
    )

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(app.run(shutdown))
        tasks.create_task(_run_ui(layout, shutdown))
        tasks.create_task(_stop_ui_on_shutdown(layout, shutdown))


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point."""
    asyncio.run(run(_parse_args(argv)))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Henry.")
    profile_kinds = [kind.name.lower() for kind in ProfileKind]
    parser.add_argument(
        "-noui",
        "--no-ui",
        action="store_true",
        help="Run with console event logging instead of the terminal UI.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("HENRY_LOG_LEVEL", "DEBUG"),
        help="Log level (env: HENRY_LOG_LEVEL).",
    )
    parser.add_argument(
        "--profile-kind",
        choices=profile_kinds,
        default=os.getenv("HENRY_PROFILE_KIND", "default").lower(),
        help="Assistant profile kind (env: HENRY_PROFILE_KIND).",
    )
    parser.add_argument(
        "--profile-name",
        default=os.getenv("HENRY_PROFILE_NAME", "Henry"),
        help="Assistant name (env: HENRY_PROFILE_NAME).",
    )
    parser.add_argument(
        "--system-language",
        default=os.getenv("HENRY_SYSTEM_LANGUAGE", "Polish"),
        help="Language used in the system prompt (env: HENRY_SYSTEM_LANGUAGE).",
    )
    parser.add_argument(
        "--wakeword-reply",
        default=os.getenv("HENRY_WAKEWORD_REPLY", "Tak, Wielmożny Panie..."),
        help="Reply after wake-word detection (env: HENRY_WAKEWORD_REPLY).",
    )
    parser.add_argument(
        "--wakeword-model",
        default=os.getenv("HENRY_WAKEWORD_MODEL", "Hey_Henree_20260406_162745.onnx"),
        help="Wake-word model path (env: HENRY_WAKEWORD_MODEL).",
    )
    parser.add_argument(
        "--voice-model",
        default=os.getenv(
            "HENRY_VOICE_MODEL", "pl/pl_PL/bass/high/pl_PL-bass-high.onnx"
        ),
        help="Piper voice model path (env: HENRY_VOICE_MODEL).",
    )
    parser.add_argument(
        "--language-model",
        default=os.getenv(
            "HENRY_LANGUAGE_MODEL", "mlx-community/Qwen3.5-9B-OptiQ-4bit"
        ),
        help="MLX language model id or path (env: HENRY_LANGUAGE_MODEL).",
    )
    config = parser.parse_args(argv)
    if config.profile_kind not in profile_kinds:
        parser.error("HENRY_PROFILE_KIND must be one of: " + ", ".join(profile_kinds))
    return config


def _configure_shutdown() -> asyncio.Event:
    shutdown = asyncio.Event()

    def request_shutdown(_: object) -> None:
        shutdown.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, request_shutdown, None)
    loop.add_signal_handler(signal.SIGTERM, request_shutdown, None)

    return shutdown


async def _run_ui(
    app: Layout,
    shutdown: asyncio.Event,
) -> None:
    try:
        await app.run_async()
    finally:
        shutdown.set()


async def _stop_ui_on_shutdown(
    app: Layout,
    shutdown: asyncio.Event,
) -> None:
    await shutdown.wait()
    app.exit()


if __name__ == "__main__":
    main()
