from henry_common import EventBus, PathLocator, Profile
from henry_speech.audio import AudioDriver
from henry_speech.capture import CaptureConfig, CaptureService, VADModel, WakeWordModel
from henry_speech.playback import PlaybackService
from henry_speech.segmentation import SegmentationConfig, SegmentationService
from henry_speech.synthesis import SynthesisModel, SynthesisService
from henry_speech.transcription import TranscriptionModel, TranscriptionService
from henry_speech.worker import SpeechWorker


def _get_audio_driver() -> AudioDriver:
    from henry_speech.audio.adapters.pyaudio import PyAudioDriver

    return PyAudioDriver()


def _get_vad_model(locator: PathLocator) -> VADModel:
    from henry_speech.capture.adapters.mlx_audio import SileroVADModel

    return SileroVADModel()


def _get_wakeword_model(locator: PathLocator, model_path: str) -> WakeWordModel:
    from henry_speech.capture.adapters.openwakeword import OpenWakeWordModel

    return OpenWakeWordModel(locator, model_path)


def _get_synthesis_model(model_path: str) -> SynthesisModel:
    from henry_speech.synthesis.adapters.piper import PiperModel

    return PiperModel(model_path)


def _get_transcription_model() -> TranscriptionModel:
    from henry_speech.transcription.adapters.mlx_audio import ParakeetTDTModel

    return ParakeetTDTModel()


async def run_speech_worker(
    locator: PathLocator,
    profile: Profile,
    event_bus: EventBus,
) -> None:

    with _get_audio_driver() as audio_driver:
        audio_input = audio_driver.get_input()
        audio_output = audio_driver.get_output()

        await SpeechWorker(
            event_bus=event_bus,
            capture_service=CaptureService(
                config=CaptureConfig(),
                audio_input=audio_input,
                vad_model=_get_vad_model(
                    locator=locator,
                ),
                wakeword_model=_get_wakeword_model(
                    locator=locator,
                    model_path=profile.wakeword.model,
                ),
            ),
            playback_service=PlaybackService(
                audio_output=audio_output,
            ),
            segmentation_service=SegmentationService(
                config=SegmentationConfig(),
            ),
            synthesis_service=SynthesisService(
                model=_get_synthesis_model(
                    model_path=profile.voice.model,
                ),
            ),
            transcription_service=TranscriptionService(
                model=_get_transcription_model(),
            ),
        ).run()
