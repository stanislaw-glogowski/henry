import asyncio
from dataclasses import dataclass
from typing import Literal

from henry_common.components import Component
from henry_common.events import EventBus, ShutdownEvent
from henry_reply.events import (
    GenerateReply,
    ReplyCompleted,
    ReplyLine,
    ReplyStarted,
)

from .audio import AudioFrame
from .capture import CaptureService, SpeechChunk
from .events import SpeechChunkCaptured, VADObserved, WakeWordObserved
from .playback import PlaybackService
from .segmentation import SegmentationService, SpeechSegment
from .synthesis import SynthesisService
from .transcription import TranscriptionChunk, TranscriptionService, TranscriptionText


@dataclass(frozen=True, slots=True)
class WorkerOptions:
    wakeword_disabled: bool = False


class Worker(Component):
    def __init__(
        self,
        event_bus: EventBus,
        capture_service: CaptureService,
        segmentation_service: SegmentationService,
        transcription_service: TranscriptionService,
        synthesis_service: SynthesisService,
        playback_service: PlaybackService,
        options: WorkerOptions | None = None,
    ) -> None:
        super().__init__()
        self._options = options if options is not None else WorkerOptions()

        self._event_bus = event_bus

        self._capture_queue: asyncio.Queue[SpeechChunk] = asyncio.Queue()
        self._capture_service = capture_service

        self._segmentation_queue: asyncio.Queue[SpeechSegment] = asyncio.Queue()
        self._segmentation_service = segmentation_service

        self._transcription_service = transcription_service

        self._synthesis_service = synthesis_service
        self._synthesis_queue: asyncio.Queue[str] = asyncio.Queue()

        self._playback_service = playback_service
        self._playback_queue: asyncio.Queue[AudioFrame] = asyncio.Queue()

        self._shutdown_event = asyncio.Event()

        self._recording = False
        self._listening = False
        self._pending_ops = 0

        self._logger.debug("INITIALIZED")

    async def run(self) -> None:
        async with (
            self._capture_service,
            self._transcription_service,
            self._synthesis_service,
            self._playback_service,
        ):
            self._logger.debug("Starting tasks")

            self._set_recording(True)
            self._set_listening(not self._options.wakeword_disabled)

            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(self._capture_loop()),
                    group.create_task(self._events_loop()),
                    group.create_task(self._segmentation_loop()),
                    group.create_task(self._transcription_loop()),
                    group.create_task(self._synthesis_loop()),
                    group.create_task(self._playback_loop()),
                ]

                await self._shutdown_event.wait()

                self._logger.debug("Canceling tasks")

                for task in tasks:
                    task.cancel()

    async def _capture_loop(self) -> None:
        async for chunk in self._capture_service.capture():
            if self._shutdown_event.is_set():
                return

            self._event_bus.publish(
                SpeechChunkCaptured.from_chunk(chunk),
                VADObserved.from_chunk(chunk),
                WakeWordObserved.from_chunk(chunk),
            )

            if not self._recording:
                continue

            if not self._listening:
                self._capture_queue.put_nowait(chunk)
                continue

            if chunk.is_wakeword:
                self._set_listening(False)
                self._inc_pending_ops()
                self._event_bus.publish(GenerateReply())

    async def _events_loop(self) -> None:
        with self._event_bus.subscribe(
            ReplyStarted,
            ReplyLine,
            ReplyCompleted,
            ShutdownEvent,
        ) as events:
            async for event in events:
                match event:
                    case ReplyStarted() if event.is_background:
                        self._inc_pending_ops()
                    case ReplyLine():
                        self._inc_pending_ops()
                        self._synthesis_queue.put_nowait(event.text)
                    case ReplyCompleted():
                        self._dec_pending_ops()
                    case ShutdownEvent():
                        self._shutdown_event.set()
                events.task_done()

    async def _segmentation_loop(self) -> None:
        while not self._shutdown_event.is_set():
            speech_chunk = await self._capture_queue.get()

            detected, speech_segment = self._segmentation_service.feed(speech_chunk)
            if not detected:
                continue

            if speech_segment:
                self._inc_pending_ops()
                self._segmentation_queue.put_nowait(speech_segment)

    async def _transcription_loop(self) -> None:
        while not self._shutdown_event.is_set():
            speech_segment = await self._segmentation_queue.get()
            try:
                async for item in self._transcription_service.transcribe(
                    speech_segment.audio
                ):
                    match item:
                        case None:
                            pass
                        case TranscriptionChunk():
                            pass
                        case TranscriptionText():
                            self._inc_pending_ops()
                            self._event_bus.publish(GenerateReply(item.content))
            finally:
                self._dec_pending_ops()
                self._segmentation_queue.task_done()

    async def _synthesis_loop(self) -> None:
        while not self._shutdown_event.is_set():
            text = await self._synthesis_queue.get()

            try:
                async for chunk in self._synthesis_service.synthesize(text):
                    self._inc_pending_ops()
                    self._playback_queue.put_nowait(chunk)
            finally:
                self._dec_pending_ops()
                self._synthesis_queue.task_done()

    async def _playback_loop(self) -> None:
        while not self._shutdown_event.is_set():
            frame = await self._playback_queue.get()

            try:
                await self._playback_service.play(frame)
            finally:
                self._dec_pending_ops()
                self._playback_queue.task_done()

    def _set_listening(self, enabled: bool) -> None:
        self._listening = enabled
        if enabled:
            self._logger.debug("Listening ENABLED")
            self._capture_service.enable_wakeword()
        else:
            self._logger.debug("Listening DISABLED")
            self._capture_service.disable_wakeword()

    def _set_recording(self, enabled: bool) -> None:
        self._recording = enabled
        if enabled:
            self._logger.debug("Recording ENABLED")
        else:
            self._logger.debug("Recording DISABLED")
            self._segmentation_service.reset()

    def _inc_pending_ops(self) -> None:
        self._update_pending_ops(1)

    def _dec_pending_ops(self) -> None:
        self._update_pending_ops(-1)

    def _update_pending_ops(self, delta: Literal[1, -1]) -> None:
        self._pending_ops += delta
        recording = self._pending_ops == 0
        if recording != self._recording:
            self._set_recording(recording)
