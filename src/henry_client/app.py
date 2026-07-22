import asyncio
from dataclasses import dataclass

from loguru import logger

from .audio.service import AudioService
from .conversation.service import ConversationService
from .events import AppEventSink
from .orchestrator import Orchestrator
from .profiles import Profile
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
        """Run the configured assistant until ``shutdown`` is requested."""
        # Heavy native adapters stay out of module import paths used by tests and tools.
        from .audio.adapters import (
            OpenWakeWordModel,
            PyAudioSession,
            PyAudioStream,
            SileroVADModel,
        )
        from .conversation.adapters import MLXLanguageModel
        from .speech.adapters import ParakeetSTTModel, PiperTTSModel

        self._logger.debug("Starting")

        with PyAudioSession() as audio_session:
            audio_input = PyAudioStream.input(audio_session)
            audio_output = PyAudioStream.output(audio_session)
            wakeword_model = OpenWakeWordModel(self._config.profile.wakeword_model)
            vad_model = SileroVADModel()

            stt_model = ParakeetSTTModel()
            tts_model = PiperTTSModel(self._config.profile.voice_model)

            language_model = MLXLanguageModel(self._config.language_model)

            async with (
                AudioService(
                    input_stream=audio_input,
                    output_stream=audio_output,
                    wakeword_model=wakeword_model,
                    vad_model=vad_model,
                ) as audio_service,
                SpeechService(
                    stt_model=stt_model,
                    tts_model=tts_model,
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
                    wakeword_reply_text=self._config.profile.wakeword_reply,
                ).run(shutdown)
