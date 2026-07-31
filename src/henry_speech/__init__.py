from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from henry_common.events import EventBus
    from henry_resources.models import ModelCatalog

    from .config import SpeechProfile, SpeechSettings

__all__ = ["run_speech_worker"]


async def run_speech_worker(
    profile: SpeechProfile,
    settings: SpeechSettings,
    model_catalog: ModelCatalog,
    event_bus: EventBus,
) -> None:
    from .audio import get_audio_driver
    from .capture import CaptureService, get_vad_model, get_wakeword_model
    from .playback import PlaybackService
    from .segmentation import SegmentationService
    from .synthesis import SynthesisService, get_tts_model
    from .transcription import TranscriptionService, get_stt_model
    from .worker import Worker

    with get_audio_driver(settings.audio) as audio_driver:
        audio_input = audio_driver.get_input()
        audio_output = audio_driver.get_output()
        vad_model = get_vad_model(
            model_catalog,
            settings.vad,
        )
        wakeword_model = get_wakeword_model(
            model_catalog,
            profile.wakeword,
            settings.wakeword,
        )
        tts_model = get_tts_model(
            profile.tts,
            settings.tts,
        )
        stt_model = get_stt_model(
            profile.stt,
            settings.stt,
        )
        await Worker(
            event_bus=event_bus,
            capture_service=CaptureService(
                audio_input=audio_input,
                vad_model=vad_model,
                wakeword_model=wakeword_model,
            ),
            playback_service=PlaybackService(
                audio_output=audio_output,
            ),
            segmentation_service=SegmentationService(
                settings=settings.segmentation,
            ),
            synthesis_service=SynthesisService(
                tts_model=tts_model,
            ),
            transcription_service=TranscriptionService(
                stt_model=stt_model,
            ),
        ).run()
