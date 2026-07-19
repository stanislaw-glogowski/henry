import asyncio
from dataclasses import dataclass

from loguru import logger

from .domain import (
    AssistantReply,
    AudioFrame,
    SpeechSegment,
    SpeechTranscription,
)
from .ports import AppEventSink, TelemetrySink
from .services import (
    CaptureService,
    ConversationService,
    PlaybackService,
    RecordingService,
    SynthesiseService,
    TranscriptionService,
)


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    input_frames_maxsize: int = 32
    output_frames_maxsize: int = 16
    speech_segments_maxsize: int = 4
    speech_transcriptions_maxsize: int = 4
    assistant_replies_maxsize: int = 4


class Orchestrator:
    def __init__(
        self,
        capture: CaptureService,
        playback: PlaybackService,
        recorder: RecordingService,
        transcription: TranscriptionService,
        conversation: ConversationService,
        synthesise: SynthesiseService,
        config: OrchestratorConfig | None = None,
    ):
        if config is None:
            config: OrchestratorConfig = OrchestratorConfig()

        assert isinstance(config, OrchestratorConfig)

        self._config = config
        self._capture = capture
        self._playback = playback
        self._recorder = recorder
        self._transcription = transcription
        self._conversation = conversation
        self._synthesise = synthesise
        self._logger = logger.bind(component="Orchestrator")

    async def run(
        self,
        events: AppEventSink,
        telemetry: TelemetrySink,
        shutdown: asyncio.Event,
    ) -> None:
        input_frames: asyncio.Queue[AudioFrame] = asyncio.Queue(
            maxsize=self._config.input_frames_maxsize,
        )
        output_frames: asyncio.Queue[AudioFrame | None] = asyncio.Queue(
            maxsize=self._config.output_frames_maxsize,
        )
        speech_segments: asyncio.Queue[SpeechSegment] = asyncio.Queue(
            maxsize=self._config.speech_segments_maxsize,
        )
        speech_transcriptions: asyncio.Queue[SpeechTranscription] = asyncio.Queue(
            maxsize=self._config.speech_transcriptions_maxsize,
        )
        assistant_replies: asyncio.Queue[AssistantReply] = asyncio.Queue(
            maxsize=self._config.assistant_replies_maxsize,
        )

        recording = asyncio.Event()
        recording.set()

        self._logger.debug("Recording ENABLED")

        self._logger.debug("Running tasks")

        async with asyncio.TaskGroup() as tasks:
            capture_task = tasks.create_task(
                self._capture.run(
                    frames=input_frames,
                    events=events,
                    telemetry=telemetry,
                    shutdown=shutdown,
                ),
            )
            playback_task = tasks.create_task(
                self._playback.run(
                    frames=output_frames,
                    recording=recording,
                    telemetry=telemetry,
                    events=events,
                ),
            )
            recorder_task = tasks.create_task(
                self._recorder.run(
                    frames=input_frames,
                    segments=speech_segments,
                    recording=recording,
                    events=events,
                    telemetry=telemetry,
                ),
            )
            transcription_task = tasks.create_task(
                self._transcription.run(
                    segments=speech_segments,
                    transcriptions=speech_transcriptions,
                    recording=recording,
                    events=events,
                    telemetry=telemetry,
                ),
            )
            conversation_task = tasks.create_task(
                self._conversation.run(
                    transcriptions=speech_transcriptions,
                    replies=assistant_replies,
                    events=events,
                    telemetry=telemetry,
                ),
            )
            synthesise_task = tasks.create_task(
                self._synthesise.run(
                    replies=assistant_replies,
                    frames=output_frames,
                    events=events,
                    telemetry=telemetry,
                ),
            )

            await shutdown.wait()

            self._logger.debug("Cancelling tasks")

            capture_task.cancel()
            playback_task.cancel()
            recorder_task.cancel()
            transcription_task.cancel()
            conversation_task.cancel()
            synthesise_task.cancel()
