import asyncio

from henry_common import AppEventSink, bind_logger

from .audio import AudioFrame
from .capture import CaptureService, SpeechChunk
from .events import SpeechChunkCaptured
from .playback import PlaybackService
from .segmentation import SegmentationService, SpeechSegment
from .synthesis import SynthesisService
from .transcription import TranscriptionChunk, TranscriptionService, TranscriptionText

type _SpeechChunks = asyncio.Queue[SpeechChunk]
type _SpeechSegments = asyncio.Queue[SpeechSegment]
type _ReplyRequests = asyncio.Queue[str]
type _Replies = asyncio.Queue[str | None]
type _PlaybackRequests = asyncio.Queue[AudioFrame | None]


class SpeechState:
    def __init__(self) -> None:
        pass


class SpeechPipeline:
    def __init__(
        self,
        capture_service: CaptureService,
        playback_service: PlaybackService,
        segmentation_service: SegmentationService,
        synthesis_service: SynthesisService,
        transcription_service: TranscriptionService,
        events: AppEventSink,
    ) -> None:
        self._capture_service = capture_service
        self._playback_service = playback_service
        self._segmentation_service = segmentation_service
        self._synthesis_service = synthesis_service
        self._transcription_service = transcription_service

        self._events = events

        self._recording = False
        self._logger = bind_logger(self)
        self._logger.debug("Pipeline INITIALIZED")

    async def run(
        self,
        shutdown: asyncio.Event,
    ) -> None:
        async with (
            self._capture_service,
            self._playback_service,
            self._synthesis_service,
            self._transcription_service,
        ):
            speech_chunks: _SpeechChunks = asyncio.Queue()
            speech_segments: _SpeechSegments = asyncio.Queue()
            reply_requests: _ReplyRequests = asyncio.Queue()
            replies: _Replies = asyncio.Queue()
            playback_requests: _PlaybackRequests = asyncio.Queue()

            self._logger.debug("Starting tasks")
            self._enable_recording()

            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(
                        self._capture_loop(
                            speech_chunks,
                            shutdown,
                        )
                    ),
                    group.create_task(
                        self._segmentation_loop(
                            speech_chunks,
                            speech_segments,
                            shutdown,
                        )
                    ),
                    group.create_task(
                        self._transcription_loop(
                            speech_segments,
                            reply_requests,
                            shutdown,
                        )
                    ),
                    group.create_task(
                        self._reply_loop(
                            reply_requests,
                            replies,
                            shutdown,
                        )
                    ),
                    group.create_task(
                        self._synthesis_loop(
                            replies,
                            playback_requests,
                            shutdown,
                        )
                    ),
                    group.create_task(
                        self._playback_loop(
                            playback_requests,
                            shutdown,
                        )
                    ),
                ]

                await shutdown.wait()

                self._logger.debug("Canceling tasks")

                for task in tasks:
                    task.cancel()

    def _enable_recording(self) -> None:
        if self._recording:
            return
        self._recording = True
        self._segmentation_service.reset()
        self._logger.debug("Recording ENABLED")

    def _disable_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._segmentation_service.reset()
        self._logger.debug("Recording DISABLED")

    async def _capture_loop(
        self,
        speech_chunks: _SpeechChunks,
        shutdown: asyncio.Event,
    ) -> None:
        async for chunk in self._capture_service.capture():
            if shutdown.is_set():
                return

            self._events.publish(
                SpeechChunkCaptured(
                    audio_len=len(chunk.audio.samples),
                    voice_detected=chunk.voice_detected,
                    voice_score=chunk.voice_score,
                    wakeword_detected=chunk.wakeword_detected,
                    wakeword_score=chunk.wakeword_score,
                )
            )

            speech_chunks.put_nowait(chunk)

    async def _segmentation_loop(
        self,
        speech_chunks: _SpeechChunks,
        speech_segments: _SpeechSegments,
        shutdown: asyncio.Event,
    ) -> None:
        while not shutdown.is_set():
            speech_chunk = await speech_chunks.get()

            detected, speech_segment = self._segmentation_service.feed(speech_chunk)
            if not detected:
                continue

            if speech_segment:
                self._disable_recording()
                speech_segments.put_nowait(speech_segment)

    async def _transcription_loop(
        self,
        speech_segments: _SpeechSegments,
        reply_requests: _ReplyRequests,
        shutdown: asyncio.Event,
    ) -> None:
        while not shutdown.is_set():
            speech_segment = await speech_segments.get()
            async for transcription in self._transcription_service.transcribe(
                speech_segment.audio
            ):
                match transcription:
                    case None:
                        self._enable_recording()
                    case TranscriptionChunk():
                        pass
                    case TranscriptionText():
                        reply_requests.put_nowait(transcription.content)

    @staticmethod
    async def _reply_loop(
        reply_requests: _ReplyRequests,
        replies: _Replies,
        shutdown: asyncio.Event,
    ) -> None:
        while not shutdown.is_set():
            request = await reply_requests.get()

            await asyncio.sleep(2.0)

            replies.put_nowait(request)
            replies.put_nowait(None)
            reply_requests.task_done()

    async def _synthesis_loop(
        self,
        replies: _Replies,
        playback_requests: _PlaybackRequests,
        shutdown: asyncio.Event,
    ) -> None:
        while not shutdown.is_set():
            reply = await replies.get()

            match reply:
                case None:
                    playback_requests.put_nowait(None)

                case str():
                    async for chunk in self._synthesis_service.synthesize(reply):
                        playback_requests.put_nowait(chunk)

    async def _playback_loop(
        self,
        playback_requests: _PlaybackRequests,
        shutdown: asyncio.Event,
    ) -> None:
        while not shutdown.is_set():
            playback_request = await playback_requests.get()

            match playback_request:
                case None:
                    self._enable_recording()

                case AudioFrame():
                    await self._playback_service.play(playback_request)
