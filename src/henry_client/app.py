import asyncio
from dataclasses import dataclass

from loguru import logger

from .audio.adapters import (
    PyAudioSession,
    PyAudioStream,
    SileroVADModel,
)
from .audio.service import AudioService
from .conversation.adapters import MLXLanguageModel
from .conversation.service import ConversationService
from .events import AppEventSink
from .orchestrator import Orchestrator
from .profiles import Profile
from .speech.adapters import OpenWakeWordModel, ParakeetSTTModel, PiperTTSModel
from .speech.service import SpeechService


@dataclass(frozen=True, slots=True)
class AppConfig:
    profile: Profile
    language_model: str


class App:
    def __init__(self, config: AppConfig, events: AppEventSink) -> None:
        self._config = config
        self._events = events
        self._logger = logger.bind(component="App")

    async def run(
        self,
        shutdown: asyncio.Event,
    ) -> None:
        self._logger.debug("Starting")
        wakeword_model = OpenWakeWordModel()
        with PyAudioSession() as audio_session, wakeword_model:
            audio_input = PyAudioStream.input(audio_session)
            audio_output = PyAudioStream.output(audio_session)
            vad_model = SileroVADModel()
            stt_model = ParakeetSTTModel()
            tts_model = PiperTTSModel(self._config.profile.voice_model)

            language_model = MLXLanguageModel(self._config.language_model)

            async with (
                AudioService(
                    input_stream=audio_input,
                    output_stream=audio_output,
                    vad_model=vad_model,
                ) as audio_service,
                SpeechService(
                    stt_model=stt_model,
                    tts_model=tts_model,
                    wakeword_model=wakeword_model,
                ) as speech_service,
                ConversationService(
                    system_prompt=self._config.profile.system_prompt,
                    model=language_model,
                ) as conversation_service,
            ):
                await Orchestrator(
                    audio=audio_service,
                    conversation=conversation_service,
                    speech=speech_service,
                    events=self._events,
                ).run(shutdown)
