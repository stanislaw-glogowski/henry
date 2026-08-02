from __future__ import annotations

import asyncio
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
    start_event: asyncio.Event | None = None,
) -> None:
    from .audio import get_audio_driver
    from .capture import CaptureService, get_vad_model, get_wakeword_model
    from .events import AudioDevicesSelected
    from .playback import PlaybackService
    from .segmentation import UtteranceSegmenter
    from .synthesis import SynthesisService, get_tts_model
    from .transcription import TranscriptionService, get_stt_model
    from .worker import Worker

    with get_audio_driver(settings.audio) as audio_driver:
        event_bus.publish(
            AudioDevicesSelected(
                driver=settings.audio.driver,
                devices=audio_driver.devices,
            )
        )
        audio_input = audio_driver.input
        audio_output = audio_driver.output
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
            profile,
            settings.tts,
        )
        stt_model = get_stt_model(
            profile,
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
            utterance_segmenter=UtteranceSegmenter(
                settings=settings.segmentation,
            ),
            synthesis_service=SynthesisService(
                tts_model=tts_model,
            ),
            transcription_service=TranscriptionService(
                stt_model=stt_model,
            ),
            start_event=start_event,
        ).run()
