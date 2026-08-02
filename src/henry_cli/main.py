import asyncio
import signal
from contextlib import suppress

from loguru import logger

from henry_common.events import EventBus, EventSubscription, ShutdownEvent
from henry_conversation import ConversationReady, run_conversation_worker
from henry_resources import LocalStore, Profile, ProfileEntry, Settings
from henry_speech import run_speech_worker
from henry_speech.events import SpeechReady

from .events import UiEventBridge, run_event_logger
from .logs import LogBuffer, configure_ui_logger
from .progress import HuggingFaceProgress, ProgressStore
from .ui import TerminalApp
from .ui.state import RuntimeInfo


def configure_shutdown(event_bus: EventBus) -> None:
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, event_bus.publish, ShutdownEvent())
    loop.add_signal_handler(signal.SIGTERM, event_bus.publish, ShutdownEvent())


async def _run_workers(
    event_bus: EventBus,
    profile: Profile,
    settings: Settings,
    store: LocalStore,
    start_event: asyncio.Event,
) -> None:
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(
            run_conversation_worker(
                event_bus,
                profile.conversation,
                settings.conversation,
                start_event,
            )
        )
        tasks.create_task(
            run_speech_worker(
                profile,
                settings.speech,
                store,
                event_bus,
                start_event,
            )
        )


async def _wait_until_ready(events: EventSubscription) -> None:
    conversation_ready = False
    speech_ready = False
    async for event in events:
        try:
            match event:
                case ConversationReady():
                    conversation_ready = True
                case SpeechReady():
                    speech_ready = True
                case ShutdownEvent():
                    raise asyncio.CancelledError
            if conversation_ready and speech_ready:
                return
        finally:
            events.task_done()


async def _start_runtime(
    event_bus: EventBus,
    profile: Profile,
    settings: Settings,
    store: LocalStore,
) -> asyncio.Task[None]:
    start_event = asyncio.Event()
    with event_bus.subscribe(
        ConversationReady,
        SpeechReady,
        ShutdownEvent,
    ) as readiness:
        backend = asyncio.create_task(
            _run_workers(event_bus, profile, settings, store, start_event),
            name="henry-runtime",
        )
        ready = asyncio.create_task(
            _wait_until_ready(readiness),
            name="henry-runtime-readiness",
        )
        done, _ = await asyncio.wait(
            (backend, ready),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if backend in done:
            ready.cancel()
            with suppress(asyncio.CancelledError):
                await ready
            await backend
        else:
            await ready
            start_event.set()
            return backend
    raise RuntimeError("Henry runtime stopped before it became ready")


def _validate_profiles(
    profiles: tuple[ProfileEntry, ...],
    settings: Settings,
) -> tuple[ProfileEntry, ...]:
    validated: list[ProfileEntry] = []
    for entry in profiles:
        if entry.profile is None:
            validated.append(entry)
            continue
        try:
            RuntimeInfo.from_runtime(entry.profile, settings)
        except Exception as error:
            validated.append(
                ProfileEntry(
                    id=entry.id,
                    name=entry.name,
                    error=f"Incompatible with current settings: {error}",
                )
            )
        else:
            validated.append(entry)
    return tuple(validated)


async def run() -> None:
    store = LocalStore()
    settings = store.load_settings()
    profiles = _validate_profiles(tuple(store.inspect_profiles()), settings)
    logs = LogBuffer()
    configure_ui_logger(logs)
    progress = ProgressStore()
    bridge = UiEventBridge()

    with EventBus() as event_bus, HuggingFaceProgress(progress):
        configure_shutdown(event_bus)
        app = TerminalApp(profiles, bridge, logs, progress)
        app_task = asyncio.create_task(app.run_async(), name="henry-ui")
        bridge_task = asyncio.create_task(bridge.run(event_bus), name="henry-ui-events")
        logger_task = asyncio.create_task(
            run_event_logger(event_bus), name="henry-logs"
        )
        backend: asyncio.Task[None] | None = None

        try:
            await asyncio.gather(app.wait_mounted(), bridge.wait_ready())
            profile = await app.select_profile()
            if profile is None:
                return
            app.configure_runtime(profile, settings)

            while True:
                await app.show_startup()
                try:
                    backend = await _start_runtime(
                        event_bus,
                        profile,
                        settings,
                        store,
                    )
                except BaseExceptionGroup as error:
                    logger.exception("Henry startup failed")
                    if not await app.wait_startup_retry(error):
                        return
                except Exception as error:
                    logger.exception("Henry startup failed")
                    if not await app.wait_startup_retry(error):
                        return
                else:
                    await app.finish_startup()
                    break

            quit_task = asyncio.create_task(
                app.wait_quit_requested(),
                name="henry-ui-quit",
            )
            done, _ = await asyncio.wait(
                (backend, app_task, quit_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if quit_task in done or app_task in done:
                event_bus.publish(ShutdownEvent())
            if backend in done:
                await backend
        finally:
            event_bus.publish(ShutdownEvent())
            if backend is not None:
                await asyncio.gather(backend, return_exceptions=True)
            app.exit()
            await asyncio.gather(
                app_task,
                bridge_task,
                logger_task,
                return_exceptions=True,
            )


def main() -> None:
    asyncio.run(run())
