import asyncio
import signal

from henry_common.events import EventBus, ShutdownEvent
from henry_reply.events import (
    GenerateReply,
    ReplyCompleted,
    ReplyLine,
    ReplyStarted,
)
from henry_resources import LocalStore
from henry_speech import run_speech_worker


def configure_shutdown() -> asyncio.Event:
    shutdown = asyncio.Event()

    def request_shutdown(_: object) -> None:
        shutdown.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, request_shutdown, None)
    loop.add_signal_handler(signal.SIGTERM, request_shutdown, None)

    return shutdown


async def main() -> None:
    local_store = LocalStore()
    shutdown = configure_shutdown()
    profile = local_store.load_profile("default")
    settings = local_store.load_settings()

    with EventBus() as event_bus:

        async def run_events() -> None:
            with event_bus.subscribe() as events:
                async for event in events:
                    match event:
                        case GenerateReply():
                            match event.text:
                                case str():
                                    event_bus.publish(
                                        ReplyStarted(),
                                        ReplyLine(text=event.text),
                                        ReplyCompleted(),
                                    )
                                case _:
                                    event_bus.publish(
                                        ReplyStarted(),
                                        ReplyLine(text="Dzień dobry!"),
                                        ReplyCompleted(),
                                    )

        async with asyncio.TaskGroup() as tasks:
            a = tasks.create_task(run_events())
            b = tasks.create_task(
                run_speech_worker(profile, settings.speech, local_store, event_bus)
            )

            await shutdown.wait()

            event_bus.publish(ShutdownEvent())

            a.cancel()
            b.cancel()


if __name__ == "__main__":
    asyncio.run(main())
