import asyncio
import signal

from henry import App, AppConfig
from henry.profiles import ProfileName, get_profile
from henry_cli.events import EventBridge
from henry_cli.logs import configure_loger
from henry_cli.telemetry import TelemetryCollector
from henry_cli.ui import Layout


async def main() -> None:
    logs = configure_loger()
    events = EventBridge()
    telemetry = TelemetryCollector()
    shutdown = _configure_shutdown()

    app = App(
        config=AppConfig(
            profile=get_profile(ProfileName.HENRY),
        ),
        events=events,
        telemetry=telemetry,
    )

    layout = Layout(
        logs=logs,
        events=events,
        telemetry=telemetry,
    )

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(
            app.run(shutdown=shutdown),
        )
        tasks.create_task(_run_ui(layout, shutdown))
        tasks.create_task(_stop_ui_on_shutdown(layout, shutdown))


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
    asyncio.run(main())
