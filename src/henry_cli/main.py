import asyncio
import signal

from henry_common.events import EventBus, ShutdownEvent
from henry_reply import run_reply_worker
from henry_reply.graph import ReplyContext
from henry_resources import LocalStore
from henry_speech import run_speech_worker


async def main() -> None:
    local_store = LocalStore()
    profile = local_store.load_profile("default")
    settings = local_store.load_settings()

    with EventBus() as event_bus:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, event_bus.publish, ShutdownEvent())
        loop.add_signal_handler(signal.SIGTERM, event_bus.publish, ShutdownEvent())

        await asyncio.gather(
            run_reply_worker(event_bus, ReplyContext.from_profile(profile)),
            run_speech_worker(profile, settings.speech, local_store, event_bus),
        )


if __name__ == "__main__":
    asyncio.run(main())
