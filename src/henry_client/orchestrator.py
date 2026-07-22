import asyncio

from loguru import logger

from .audio import AudioFrame
from .audio.service import AudioService
from .conversation import MessageChunk, MessageLine
from .conversation.service import ConversationService
from .events import AppEventSink, AudioCaptured, AudioPlayed, PipelineStageChanged
from .pipeline import PipelineStage, PipelineStageStatus
from .speech.service import SpeechService


# time.perf_counter()
class Orchestrator:
    def __init__(
        self,
        audio: AudioService,
        conversation: ConversationService,
        speech: SpeechService,
        events: AppEventSink,
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
        self._recording = asyncio.Event()

    async def run(
        self,
        shutdown: asyncio.Event,
    ) -> None:
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

            self._set_recording(True)

            await shutdown.wait()

            self._logger.debug("Cancelling tasks")

            for task in tasks:
                task.cancel()

    def _set_recording(self, recording: bool) -> None:
        if recording == self._recording.is_set():
            return

        if recording:
            self._recording.set()

            self._logger.debug("Recording STARTED")
            self._events.publish(
                PipelineStageChanged(
                    PipelineStage.RECORDING, PipelineStageStatus.STARTED
                )
            )
        else:
            self._recording.clear()

            self._logger.debug("Recording STOPPED")
            self._events.publish(
                PipelineStageChanged(
                    PipelineStage.RECORDING, PipelineStageStatus.COMPLETED
                )
            )

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
                    vad_score=chunk.vad_score,
                    is_speech=chunk.is_speech,
                )
            )

            if not self._recording.is_set():
                continue

            segment_ended, segment = self._speech.detect(chunk)
            if not segment_ended or segment is None:
                continue

            self._set_recording(False)

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
                    self._set_recording(True)
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
                if line is None:
                    if playing:
                        playing = False
                        self._logger.debug("Playback COMPLETED")
                        self._events.publish(
                            PipelineStageChanged(
                                PipelineStage.PLAYBACK, PipelineStageStatus.COMPLETED
                            )
                        )
                    self._set_recording(True)
                else:
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

            finally:
                self._replaying.task_done()
