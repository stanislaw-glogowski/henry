import asyncio
from enum import Enum, auto

from loguru import logger

from .audio import AudioFrame
from .audio.service import AudioService
from .conversation import MessageChunk, MessageLine
from .conversation.service import ConversationService
from .events import AppEventSink, AudioCaptured, AudioPlayed, PipelineStageChanged
from .pipeline import PipelineStage, PipelineStageStatus
from .speech.service import SpeechService


class ListeningMode(Enum):
    UNKNOWN = auto()
    WAKEWORD = auto()
    UTTERANCE = auto()
    PAUSED = auto()


class Orchestrator:
    _WAKEWORD_REPLY_TRIGGER = "WAKEWORD_REPLY"
    _WAKEWORD_REPLY_START_DELAY_SECONDS = 0.5
    _WAKEWORD_REPLY_END_DELAY_SECONDS = 0.5

    def __init__(
        self,
        audio: AudioService,
        conversation: ConversationService,
        speech: SpeechService,
        events: AppEventSink,
        wakeword_reply_text: str | None = None,
    ):
        self._audio = audio
        self._conversation = conversation
        self._speech = speech
        self._events = events
        self._logger = logger.bind(component="Orchestrator")

        self._stages: dict[PipelineStage, PipelineStageStatus]

        self._segments: asyncio.Queue[AudioFrame] = asyncio.Queue()
        self._processing: asyncio.Queue[str] = asyncio.Queue()
        self._replaying: asyncio.Queue[str | None] = asyncio.Queue()

        self._wakeword_reply_text = wakeword_reply_text
        self._wakeword_reply_frames: list[AudioFrame] = []

        self._listening_mode = ListeningMode.UNKNOWN

    async def run(
        self,
        shutdown: asyncio.Event,
    ) -> None:
        await self._preload_wakeword_reply()

        async with asyncio.TaskGroup() as group:
            self._logger.debug("Running tasks")

            tasks = [
                group.create_task(
                    self._capture_loop(shutdown),
                ),
                group.create_task(
                    self._transcribe_loop(shutdown),
                ),
                group.create_task(
                    self._processing_loop(shutdown),
                ),
                group.create_task(
                    self._replaying_loop(shutdown),
                ),
            ]

            self._set_listening_mode(ListeningMode.WAKEWORD)

            await shutdown.wait()

            self._logger.debug("Cancelling tasks")

            for task in tasks:
                task.cancel()

    async def _preload_wakeword_reply(self) -> None:
        if not self._wakeword_reply_text:
            return

        self._logger.debug("Preloading wakeword reply")

        async for frame in self._speech.synthesize(self._wakeword_reply_text):
            self._wakeword_reply_frames.append(frame)

    def _set_listening_mode(self, mode: ListeningMode) -> bool:
        if mode == self._listening_mode:
            return False

        match self._listening_mode:
            case ListeningMode.WAKEWORD:
                self._logger.debug("Listening COMPLETED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.LISTENING, PipelineStageStatus.COMPLETED
                    )
                )

            case ListeningMode.UTTERANCE:
                self._logger.debug("Recording COMPLETED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.RECORDING, PipelineStageStatus.COMPLETED
                    )
                )

        match mode:
            case ListeningMode.WAKEWORD:
                self._logger.debug("Listening STARTED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.LISTENING, PipelineStageStatus.STARTED
                    )
                )

            case ListeningMode.UTTERANCE:
                self._logger.debug("Recording STARTED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.RECORDING, PipelineStageStatus.STARTED
                    )
                )

        self._listening_mode = mode
        return True

    async def _capture_loop(
        self,
        shutdown: asyncio.Event,
    ) -> None:
        chunks = self._audio.read()

        self._events.publish(
            PipelineStageChanged(PipelineStage.CAPTURE, PipelineStageStatus.STARTED)
        )

        async for chunk in chunks:
            if shutdown.is_set():
                return

            self._events.publish(
                AudioCaptured(
                    samples_count=len(chunk.samples),
                    speech_score=chunk.speech_score,
                    speech_detected=chunk.speech_detected,
                    wakeword_score=chunk.wakeword_score,
                    wakeword_detected=chunk.wakeword_detected,
                )
            )

            match self._listening_mode:
                case ListeningMode.WAKEWORD:
                    if chunk.wakeword_detected:
                        self._set_listening_mode(ListeningMode.PAUSED)
                        self._replaying.put_nowait(self._WAKEWORD_REPLY_TRIGGER)
                case ListeningMode.UTTERANCE:
                    segment_ended, segment = self._speech.detect(chunk)
                    if segment_ended and segment is not None:
                        self._set_listening_mode(ListeningMode.PAUSED)
                        self._segments.put_nowait(segment)

        self._events.publish(
            PipelineStageChanged(PipelineStage.CAPTURE, PipelineStageStatus.COMPLETED)
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
                        PipelineStage.TRANSCRIPTION, PipelineStageStatus.STARTED
                    )
                )

                text = await self._speech.transcribe(segment)

                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.TRANSCRIPTION, PipelineStageStatus.COMPLETED
                    )
                )

                if text is None:
                    self._logger.debug("Transcribing COMPLETED: text=None")
                    self._set_listening_mode(ListeningMode.UTTERANCE)
                else:
                    self._logger.debug("Transcribing COMPLETED: text='{}'", text)
                    await self._processing.put(text)
            finally:
                self._segments.task_done()

    async def _processing_loop(
        self,
        shutdown: asyncio.Event,
    ) -> None:
        while not shutdown.is_set():
            text = await self._processing.get()
            try:
                self._logger.debug("Processing STARTED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.PROCESSING, PipelineStageStatus.STARTED
                    )
                )

                replies = self._conversation.generate_reply(text)
                async for reply in replies:
                    match reply:
                        case MessageLine():
                            self._logger.debug(
                                "Generated line: content='{}'",
                                reply.content,
                            )
                            self._replaying.put_nowait(reply.content)

                        case MessageChunk():
                            self._logger.trace(
                                "Generated chunk: content='{}'",
                                reply.content,
                            )

                        case None:
                            self._logger.trace(
                                "Generated reply: content=NONE",
                            )
                        case str():
                            self._logger.trace(
                                "Generated reply: content='{}'",
                                reply,
                            )

                self._logger.debug("Processing COMPLETED")
                self._events.publish(
                    PipelineStageChanged(
                        PipelineStage.PROCESSING, PipelineStageStatus.COMPLETED
                    )
                )
                self._replaying.put_nowait(None)
            finally:
                self._processing.task_done()

    async def _replaying_loop(
        self,
        shutdown: asyncio.Event,
    ) -> None:
        playing = False
        while not shutdown.is_set():
            line = await self._replaying.get()
            try:
                match line:
                    case self._WAKEWORD_REPLY_TRIGGER:
                        await asyncio.sleep(self._WAKEWORD_REPLY_START_DELAY_SECONDS)
                        for frame in self._wakeword_reply_frames:
                            await self._audio.write(frame)
                        await asyncio.sleep(self._WAKEWORD_REPLY_END_DELAY_SECONDS)
                        self._set_listening_mode(ListeningMode.UTTERANCE)

                    case str():
                        self._logger.debug("Synthesising STARTED")
                        self._events.publish(
                            PipelineStageChanged(
                                PipelineStage.SYNTHESIS,
                                PipelineStageStatus.STARTED,
                            )
                        )

                        frames = self._speech.synthesize(line)

                        async for frame in frames:
                            self._logger.debug(
                                "Synthesis: samples=[{}]",
                                len(frame.samples),
                            )
                            if not playing:
                                playing = True
                                self._logger.debug("Playback STARTED")
                                self._events.publish(
                                    PipelineStageChanged(
                                        PipelineStage.PLAYBACK,
                                        PipelineStageStatus.STARTED,
                                    )
                                )

                            await self._audio.write(frame)
                            self._events.publish(
                                AudioPlayed(
                                    samples_count=len(frame.samples),
                                )
                            )

                        self._logger.debug("Synthesising COMPLETED")
                        self._events.publish(
                            PipelineStageChanged(
                                PipelineStage.SYNTHESIS,
                                PipelineStageStatus.COMPLETED,
                            )
                        )
                    case None:
                        if playing:
                            playing = False
                            self._logger.debug("Playback COMPLETED")
                            self._events.publish(
                                PipelineStageChanged(
                                    PipelineStage.PLAYBACK,
                                    PipelineStageStatus.COMPLETED,
                                )
                            )
                        self._set_listening_mode(ListeningMode.UTTERANCE)
            finally:
                self._replaying.task_done()
