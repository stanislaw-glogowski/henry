import asyncio
from dataclasses import dataclass
from enum import Enum, auto

from loguru import logger

from .audio import AudioFrame
from .audio.service import AudioService
from .config import VADConfig, WakeWordConfig
from .events import AppEventSink, AudioCaptured, AudioPlayed, PipelineStageChanged
from .pipeline import PipelineStage, PipelineStageStatus
from .reply import ReplyChunk, ReplyLine, ReplyRequest, ReplySignal, ReplyText
from .reply.service import ReplyService
from .speech.service import SpeechService


class ListeningMode(Enum):
    UNKNOWN = auto()
    WAKEWORD = auto()
    UTTERANCE = auto()
    PAUSED = auto()


@dataclass(frozen=True, slots=True)
class _PlaybackEnd:
    """Mark a reply boundary and its post-playback listening delay."""

    delay: float = 0.0


class Orchestrator:
    """Coordinate listening modes and the audio, speech, and reply pipeline."""

    def __init__(
        self,
        audio: AudioService,
        reply: ReplyService,
        speech: SpeechService,
        events: AppEventSink,
        vad_config: VADConfig | None = None,
        wakeword_config: WakeWordConfig | None = None,
        activation_end_delay: float = 0.5,
    ) -> None:
        if activation_end_delay < 0:
            raise ValueError("Activation end delay cannot be negative")

        self._audio = audio
        self._reply = reply
        self._speech = speech
        self._events = events
        self._vad_config = vad_config or VADConfig()
        self._wakeword_config = wakeword_config or WakeWordConfig()
        self._activation_end_delay = activation_end_delay
        self._logger = logger.bind(component="Orchestrator")

        self._segments: asyncio.Queue[AudioFrame] = asyncio.Queue()
        self._requests: asyncio.Queue[ReplyRequest] = asyncio.Queue()
        self._playback: asyncio.Queue[str | _PlaybackEnd] = asyncio.Queue()
        self._listening_mode = ListeningMode.UNKNOWN

    async def run(
        self,
        shutdown: asyncio.Event,
    ) -> None:
        """Run the assistant pipeline until ``shutdown`` is requested."""
        async with asyncio.TaskGroup() as group:
            self._logger.debug("Running tasks")
            tasks = [
                group.create_task(self._capture_loop(shutdown)),
                group.create_task(self._transcribe_loop(shutdown)),
                group.create_task(self._reply_loop(shutdown)),
                group.create_task(self._playback_loop(shutdown)),
            ]

            self._set_listening_mode(ListeningMode.WAKEWORD)
            await shutdown.wait()

            self._logger.debug("Cancelling tasks")
            for task in tasks:
                task.cancel()

    def _set_listening_mode(self, mode: ListeningMode) -> bool:
        if mode is self._listening_mode:
            return False

        match self._listening_mode:
            case ListeningMode.WAKEWORD:
                self._audio.disable_wakeword()
                self._logger.debug("Listening COMPLETED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.LISTENING,
                        PipelineStageStatus.COMPLETED,
                    )
                )
            case ListeningMode.UTTERANCE:
                self._logger.debug("Recording COMPLETED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.RECORDING,
                        PipelineStageStatus.COMPLETED,
                    )
                )

        match mode:
            case ListeningMode.WAKEWORD:
                self._audio.reset_wakeword()
                self._audio.enable_wakeword()
                self._logger.debug("Listening STARTED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.LISTENING,
                        PipelineStageStatus.STARTED,
                    )
                )
            case ListeningMode.UTTERANCE:
                self._logger.debug("Recording STARTED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.RECORDING,
                        PipelineStageStatus.STARTED,
                    )
                )

        self._listening_mode = mode
        return True

    async def _capture_loop(
        self,
        shutdown: asyncio.Event,
    ) -> None:
        self._events.publish(
            PipelineStageChanged(
                PipelineStage.CAPTURE,
                PipelineStageStatus.STARTED,
            )
        )

        try:
            async for chunk in self._audio.capture():
                if shutdown.is_set():
                    return

                speech_detected = chunk.vad_score >= self._vad_config.threshold
                wakeword_detected: bool | None = None
                if chunk.wakeword_score is not None:
                    wakeword_detected = (
                        speech_detected
                        and chunk.wakeword_score >= self._wakeword_config.threshold
                    )

                self._events.publish(
                    AudioCaptured(
                        samples_count=len(chunk.frame.samples),
                        speech_score=chunk.vad_score,
                        speech_detected=speech_detected,
                        wakeword_score=chunk.wakeword_score,
                        wakeword_detected=wakeword_detected,
                    )
                )

                match self._listening_mode:
                    case ListeningMode.WAKEWORD if wakeword_detected:
                        self._set_listening_mode(ListeningMode.PAUSED)
                        self._requests.put_nowait(ReplySignal.ACTIVATION)
                    case ListeningMode.UTTERANCE:
                        segment_ended, segment = self._speech.segment(
                            chunk.frame,
                            speech_detected,
                        )
                        if segment_ended and segment is not None:
                            self._set_listening_mode(ListeningMode.PAUSED)
                            self._segments.put_nowait(segment)
        finally:
            self._events.publish(
                PipelineStageChanged(
                    PipelineStage.CAPTURE,
                    PipelineStageStatus.COMPLETED,
                )
            )

    async def _transcribe_loop(
        self,
        shutdown: asyncio.Event,
    ) -> None:
        while not shutdown.is_set():
            segment = await self._segments.get()
            try:
                self._logger.debug("Transcribing STARTED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.TRANSCRIPTION,
                        PipelineStageStatus.STARTED,
                    )
                )

                text = await self._speech.transcribe(segment)

                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.TRANSCRIPTION,
                        PipelineStageStatus.COMPLETED,
                    )
                )

                if text is None:
                    self._logger.debug("Transcribing COMPLETED: text=None")
                    self._set_listening_mode(ListeningMode.UTTERANCE)
                else:
                    self._logger.debug("Transcribing COMPLETED: text='{}'", text)
                    self._requests.put_nowait(text)
            finally:
                self._segments.task_done()

    async def _reply_loop(
        self,
        shutdown: asyncio.Event,
    ) -> None:
        while not shutdown.is_set():
            request = await self._requests.get()
            try:
                self._logger.debug("Processing STARTED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.PROCESSING,
                        PipelineStageStatus.STARTED,
                    )
                )

                async for reply in self._reply.reply(request):
                    match reply:
                        case ReplyLine():
                            self._logger.debug(
                                "Generated line: content='{}'",
                                reply.content,
                            )
                            self._playback.put_nowait(reply.content)
                        case ReplyChunk():
                            self._logger.trace(
                                "Generated chunk: content='{}'",
                                reply.content,
                            )
                        case ReplyText():
                            self._logger.trace(
                                "Generated reply: content='{}'",
                                reply.content,
                            )

                self._logger.debug("Processing COMPLETED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.PROCESSING,
                        PipelineStageStatus.COMPLETED,
                    )
                )
                self._playback.put_nowait(
                    _PlaybackEnd(
                        delay=(
                            self._activation_end_delay
                            if request is ReplySignal.ACTIVATION
                            else 0.0
                        )
                    )
                )
            finally:
                self._requests.task_done()

    async def _playback_loop(
        self,
        shutdown: asyncio.Event,
    ) -> None:
        playing = False
        while not shutdown.is_set():
            item = await self._playback.get()
            try:
                if isinstance(item, _PlaybackEnd):
                    if playing:
                        playing = False
                        self._logger.debug("Playback COMPLETED")
                        self._events.publish(
                            PipelineStageChanged(
                                PipelineStage.PLAYBACK,
                                PipelineStageStatus.COMPLETED,
                            )
                        )
                    if item.delay:
                        await asyncio.sleep(item.delay)
                    self._set_listening_mode(ListeningMode.UTTERANCE)
                    continue

                text = item
                self._logger.debug("Synthesising STARTED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.SYNTHESIS,
                        PipelineStageStatus.STARTED,
                    )
                )

                async for frame in self._speech.synthesize(text):
                    if not playing:
                        playing = True
                        self._logger.debug("Playback STARTED")
                        self._events.publish(
                            PipelineStageChanged(
                                PipelineStage.PLAYBACK,
                                PipelineStageStatus.STARTED,
                            )
                        )

                    await self._audio.playback(frame)
                    self._events.publish(AudioPlayed(samples_count=len(frame.samples)))

                self._logger.debug("Synthesising COMPLETED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.SYNTHESIS,
                        PipelineStageStatus.COMPLETED,
                    )
                )
            finally:
                self._playback.task_done()
