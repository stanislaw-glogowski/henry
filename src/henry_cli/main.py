import asyncio
import signal

from henry_cli.events import EventBridge
from henry_cli.logs import LogBuffer
from henry_cli.ui import Layout
from henry_client import App, AppConfig, Profile, ProfileKind

LOG_LEVEL = "DEBUG"

HENRY_PROFILE = Profile.build(
    kind=ProfileKind.DEFAULT,
    name="Henry",
    system_language="Polish",
    wakeword_reply="Tak, Wielmożny Panie...",
    wakeword_model="Hey_Henree_20260406_162745.onnx",
    voice_model="pl/pl_PL/bass/high/pl_PL-bass-high.onnx",
)


APP_CONFIG = AppConfig(
    profile=HENRY_PROFILE,
    language_model="mlx-community/Qwen3.5-9B-OptiQ-4bit",
)


async def run() -> None:
    """Run the terminal UI and assistant in one asyncio event loop."""
    logs = LogBuffer(LOG_LEVEL)
    events = EventBridge()
    shutdown = _configure_shutdown()

    app = App(
        config=APP_CONFIG,
        events=events,
    )

    layout = Layout(
        logs=logs,
        events=events,
    )

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(app.run(shutdown))
        tasks.create_task(_run_ui(layout, shutdown))
        tasks.create_task(_stop_ui_on_shutdown(layout, shutdown))


def main() -> None:
    """Console-script entry point."""
    asyncio.run(run())


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
