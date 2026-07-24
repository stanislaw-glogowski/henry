import asyncio
from dataclasses import dataclass

from loguru import logger

from .audio.service import AudioService
from .config import VADConfig, WakeWordConfig
from .events import AppEventSink
from .orchestrator import Orchestrator
from .profiles import Profile
from .reply.service import ReplyService
from .speech.service import SpeechService


@dataclass(frozen=True, slots=True)
class AppConfig:
    profile: Profile
    language_model: str
    vad: VADConfig = VADConfig()
    wakeword: WakeWordConfig = WakeWordConfig()
    max_empty_segments: int = 3


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
        from .reply.adapters.mlx_lm import MLXResponder, MLXResponderConfig
        from .speech.adapters import ParakeetSTTModel, PiperTTSModel

        self._logger.debug("Starting")

        with PyAudioSession() as audio_session:
            audio_input = PyAudioStream.input(audio_session)
            audio_output = PyAudioStream.output(audio_session)
            wakeword_model = OpenWakeWordModel(
                self._config.wakeword.model_path or self._config.profile.wakeword_model
            )
            vad_model = SileroVADModel()

            stt_model = ParakeetSTTModel()
            tts_model = PiperTTSModel(self._config.profile.voice_model)

            responder = MLXResponder(
                config=MLXResponderConfig(
                    model_id=self._config.language_model,
                    system_prompt=self._config.profile.system_prompt,
                    activation_text=(
                        self._config.wakeword.reply_message
                        if self._config.wakeword.reply_message is not None
                        else self._config.profile.wakeword_reply
                    ),
                ),
            )

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
                ReplyService(
                    responder=responder,
                ) as reply_service,
            ):
                await Orchestrator(
                    audio=audio_service,
                    reply=reply_service,
                    speech=speech_service,
                    events=self._events,
                    vad_config=self._config.vad,
                    wakeword_config=self._config.wakeword,
                    max_empty_segments=self._config.max_empty_segments,
                ).run(shutdown)
