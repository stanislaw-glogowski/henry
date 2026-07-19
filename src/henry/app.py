import asyncio
from dataclasses import dataclass

from loguru import logger

from .adapters.audio import PyAudioInput, PyAudioOutput, PyAudioSession
from .adapters.llm import MlxLLModel
from .adapters.stt import ParakeetTranscriber
from .adapters.tts import PiperSynthesizer
from .adapters.vad import SileroVad
from .domain import AudioFormat, Profile
from .events import AppInitialized
from .orchestrator import Orchestrator
from .ports import AppEventSink, TelemetrySink
from .services import (
    CaptureService,
    ConversationConfig,
    ConversationService,
    PlaybackService,
    RecordingService,
    SynthesiseService,
    TranscriptionService,
)

DEFAULT_LLMODEL_ID = "mlx-community/Qwen3.5-4B-MLX-4bit"
DEFAULT_LLMODEL_MAX_TOKENS = 128
DEFAULT_VAD_THRESHOLD = 0.5

AUDIO_INPUT_FORMAT = AudioFormat(sample_rate=16_000, channels=1)
AUDIO_INPUT_FRAMES_PER_BUFFER = 512

OUTPUT_FRAMES_PER_BUFFER = 512
LLMODEL_MAX_TOKENS = 256


@dataclass
class AppConfig:
    profile: Profile
    llmodel_id: str = DEFAULT_LLMODEL_ID
    llmodel_max_tokens: int = DEFAULT_LLMODEL_MAX_TOKENS
    vad_threshold: int = DEFAULT_VAD_THRESHOLD


class App:
    def __init__(
        self,
        config: AppConfig,
        events: AppEventSink,
        telemetry: TelemetrySink,
    ) -> None:
        self._config = config
        self._events = events
        self._telemetry = telemetry

        logger.debug("Initialized")

        events.publish(
            AppInitialized(
                profile_name=config.profile.name,
                vad_threshold=config.vad_threshold,
            )
        )

    async def run(self, shutdown: asyncio.Event) -> None:
        logger.debug("Setting up")

        audio_session = PyAudioSession()
        vad = SileroVad(
            threshold=self._config.vad_threshold,
        )

        with audio_session, vad:
            audio_input = PyAudioInput(
                audio_session,
                audio_format=AUDIO_INPUT_FORMAT,
                frames_per_buffer=AUDIO_INPUT_FRAMES_PER_BUFFER,
            )
            audio_output = PyAudioOutput(
                audio_session,
                frames_per_buffer=OUTPUT_FRAMES_PER_BUFFER,
            )

            transcriber = ParakeetTranscriber()
            language_model = MlxLLModel(
                self._config.llmodel_id,
                self._config.llmodel_max_tokens,
            )
            synthesizer = PiperSynthesizer(self._config.profile.voice_path)

            async with (
                audio_input,
                audio_output,
                transcriber,
                language_model,
                synthesizer,
            ):
                orch = Orchestrator(
                    capture=CaptureService(audio_input),
                    playback=PlaybackService(audio_output),
                    recorder=RecordingService(vad),
                    transcription=TranscriptionService(transcriber),
                    conversation=ConversationService(
                        language_model,
                        ConversationConfig(
                            system_prompt=self._config.profile.system_prompt,
                        ),
                    ),
                    synthesise=SynthesiseService(synthesizer),
                )

                logger.debug("Running")

                await orch.run(
                    shutdown=shutdown,
                    events=self._events,
                    telemetry=self._telemetry,
                )
