import asyncio
import signal

from henry_common.events import EventBus, ShutdownEvent
from henry_conversation import run_conversation_worker
from henry_conversation.graph import ConversationContext
from henry_resources import LocalStore
from henry_speech import run_speech_worker

from .events import run_event_logger
from .logger import configure_console_logger


def configure_shutdown(event_bus: EventBus) -> None:
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, event_bus.publish, ShutdownEvent())
    loop.add_signal_handler(signal.SIGTERM, event_bus.publish, ShutdownEvent())


async def run() -> None:
    configure_console_logger()
    local_store = LocalStore()
    profile = local_store.load_profile("default")
    settings = local_store.load_settings()
    conversation_context = ConversationContext.from_profile(
        profile.conversation,
    )

    with EventBus() as event_bus:
        configure_shutdown(event_bus)

        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(run_event_logger(event_bus))
            tasks.create_task(run_conversation_worker(event_bus, conversation_context))
            tasks.create_task(
                run_speech_worker(profile, settings.speech, local_store, event_bus)
            )


def main() -> None:
    asyncio.run(run())
