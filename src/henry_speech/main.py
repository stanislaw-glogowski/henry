import asyncio
import signal

from loguru import logger

from henry_common import EventBus, PathLocator, load_profile
from henry_common.events import ShutdownEvent
from henry_reply.events import (
    GenerateReply,
    ReplyCompleted,
    ReplyLine,
    ReplyStarted,
)
from henry_speech import run_speech_worker
from henry_speech.audio.adapters.pyaudio import PyAudioDriver
from henry_speech.capture import CaptureConfig, CaptureService
from henry_speech.capture.adapters.mlx_audio import SileroVADModel
from henry_speech.capture.adapters.openwakeword import OpenWakeWordModel
from henry_speech.playback import PlaybackService
from henry_speech.segmentation import SegmentationConfig, SegmentationService
from henry_speech.synthesis import SynthesisService
from henry_speech.synthesis.adapters.piper import PiperModel
from henry_speech.transcription import TranscriptionService
from henry_speech.transcription.adapters.mlx_audio import ParakeetTDTModel


def configure_shutdown() -> asyncio.Event:
    shutdown = asyncio.Event()

    def request_shutdown(_: object) -> None:
        shutdown.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, request_shutdown, None)
    loop.add_signal_handler(signal.SIGTERM, request_shutdown, None)

    return shutdown


async def main() -> None:
    locator = PathLocator()
    shutdown = configure_shutdown()
    profile = load_profile(locator, "default")

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
            b = tasks.create_task(run_speech_worker(locator, profile, event_bus))

            await shutdown.wait()

            event_bus.publish(ShutdownEvent())

            a.cancel()
            b.cancel()


if __name__ == "__main__":
    asyncio.run(main())
