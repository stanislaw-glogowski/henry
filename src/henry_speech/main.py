import asyncio
import signal

from henry_common import AppEvent, AppEventSink, PathLocator
from henry_speech.audio.adapters.pyaudio import PyAudioDriver
from henry_speech.capture import CaptureConfig, CaptureService
from henry_speech.capture.adapters.mlx_audio import SileroVADModel
from henry_speech.capture.adapters.openwakeword import OpenWakeWordModel
from henry_speech.pipeline import SpeechPipeline
from henry_speech.playback import PlaybackService
from henry_speech.segmentation import SegmentationConfig, SegmentationService
from henry_speech.synthesis import SynthesisService
from henry_speech.synthesis.adapters.piper import PiperModel
from henry_speech.transcription import TranscriptionService
from henry_speech.transcription.adapters.mlx_audio import ParakeetTDTModel


class EventLogger(AppEventSink):
    def publish(self, *events: AppEvent) -> None:
        pass


def configure_shutdown() -> asyncio.Event:
    shutdown = asyncio.Event()

    def request_shutdown(_: object) -> None:
        shutdown.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, request_shutdown, None)
    loop.add_signal_handler(signal.SIGTERM, request_shutdown, None)

    return shutdown


async def main() -> None:
    shutdown = configure_shutdown()
    locator = PathLocator()

    with PyAudioDriver() as audio_session:
        audio_input = audio_session.get_input()
        audio_output = audio_session.get_output()

        pipeline = SpeechPipeline(
            capture_service=CaptureService(
                config=CaptureConfig(),
                audio_input=audio_input,
                vad_model=SileroVADModel(),
                wakeword_model=OpenWakeWordModel(
                    locator=locator,
                    model_path="alexa_v0.1.onnx",
                ),
            ),
            playback_service=PlaybackService(
                audio_output=audio_output,
            ),
            segmentation_service=SegmentationService(
                config=SegmentationConfig(),
            ),
            synthesis_service=SynthesisService(
                model=PiperModel(
                    model_path="pl/pl_PL/bass/high/pl_PL-bass-high.onnx",
                ),
            ),
            transcription_service=TranscriptionService(
                model=ParakeetTDTModel(),
            ),
            events=EventLogger(),
        )

        await pipeline.run(shutdown)


if __name__ == "__main__":
    asyncio.run(main())
