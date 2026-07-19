import asyncio
import signal
import sys

from loguru import logger

from henry_core import AudioService, Orchestrator, SpeechService
from henry_core.audio.adapters import PyAudioManager
from henry_core.events import AppEvent, AppEventSink
from henry_core.speech.adapters import SileroVAD


class EventLogger(AppEventSink):
    def publish(self, *events: AppEvent) -> None:
        for event in events:
            logger.debug("Event: {}", event)


def configure_loger(level: str = "TRACE") -> None:
    logger.remove()
    logger.configure(
        extra={"component": "App"},
    )
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <6}</level> | "
            "<magenta>@{extra[component]}</magenta> : "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


def configure_shutdown() -> asyncio.Event:
    shutdown = asyncio.Event()

    def signal_handler(_: object) -> None:
        shutdown.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler, None)
    loop.add_signal_handler(signal.SIGTERM, signal_handler, None)

    return shutdown


async def main() -> None:
    configure_loger()
    shutdown = configure_shutdown()

    events = EventLogger()

    with PyAudioManager() as audio_streams, SileroVAD() as vad:
        async with (
            AudioService(
                streams=audio_streams,
                events=events,
            ) as audio_service,
            SpeechService(
                vad=vad,
                events=events,
            ) as speech_service,
        ):
            await Orchestrator(
                audio=audio_service,
                speech=speech_service,
                events=events,
            ).run(shutdown)


if __name__ == "__main__":
    asyncio.run(main())
