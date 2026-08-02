import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from time import perf_counter_ns

from henry_common.components import Component
from henry_common.events import EventBus, ShutdownEvent
from henry_conversation.events import (
    CancelReply,
    ConversationActivated,
    GenerateReply,
    PhraseId,
    ReplyGenerationCompleted,
    ReplyGenerationStarted,
    ReplyId,
    ReplyPhrase,
    UserTurn,
)

from .audio import AudioFrame, AudioPlaybackOutcome
from .capture import CaptureService, SpeechChunk
from .events import (
    InteractionStage,
    InteractionTimingObserved,
    ReplyPhraseDelivered,
    ReplyPhrasePlaybackStarted,
    SpeechChunkCaptured,
    SpeechReady,
    TranscriptionProgressObserved,
    TurnId,
    UserTurnCommitted,
    VADObserved,
    VoiceSessionMode,
    VoiceSessionModeChanged,
    WakeWordObserved,
)
from .playback import PlaybackService
from .segmentation import SpeechSegment, UtteranceSegmenter
from .synthesis import SynthesisService
from .transcription import (
    TranscriptionChunk,
    TranscriptionService,
    TranscriptionText,
    TurnEndpointDetector,
)


@dataclass(frozen=True, slots=True)
class WorkerOptions:
    """Frame-based voice-session thresholds for the 16 kHz capture stream."""

    wakeword_disabled: bool = False
    barge_in_speech_frames: int = 6
    continuation_silence_frames: int = 38

    def __post_init__(self) -> None:
        if self.barge_in_speech_frames <= 0:
            raise ValueError(
                "barge_in_speech_frames must be positive; "
                f"got {self.barge_in_speech_frames}"
            )
        if self.continuation_silence_frames <= 0:
            raise ValueError(
                "continuation_silence_frames must be positive; "
                f"got {self.continuation_silence_frames}"
            )


@dataclass(frozen=True, slots=True)
class _SynthesisRequest:
    reply_id: ReplyId
    phrase_id: PhraseId
    text: str


@dataclass(frozen=True, slots=True)
class _ReplyAudioFrame:
    reply_id: ReplyId
    phrase_id: PhraseId
    frame: AudioFrame


@dataclass(frozen=True, slots=True)
class _ReplyPhraseBoundary:
    reply_id: ReplyId
    phrase_id: PhraseId
    text: str


type _PlaybackItem = _ReplyAudioFrame | _ReplyPhraseBoundary


class _PlaybackControl(Enum):
    DUCK = auto()
    RESTORE = auto()
    INTERRUPT = auto()


class Worker(Component):
    _TELEMETRY_INTERVAL_FRAMES = 3

    def __init__(
        self,
        event_bus: EventBus,
        capture_service: CaptureService,
        utterance_segmenter: UtteranceSegmenter,
        transcription_service: TranscriptionService,
        synthesis_service: SynthesisService,
        playback_service: PlaybackService,
        options: WorkerOptions | None = None,
        start_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__()
        self._options = options if options is not None else WorkerOptions()

        self._event_bus = event_bus

        self._capture_queue: asyncio.Queue[SpeechChunk] = asyncio.Queue()
        self._capture_service = capture_service

        self._segmentation_queue: asyncio.Queue[SpeechSegment] = asyncio.Queue()
        self._utterance_segmenter = utterance_segmenter

        self._transcription_service = transcription_service
        self._turn_endpoint_detector = TurnEndpointDetector()
        self._pending_turn_text = ""
        self._pending_turn_silence_frames = 0
        self._pending_turn_continuation = False
        self._turn_id: TurnId | None = None
        self._turn_sequence = 0

        self._synthesis_service = synthesis_service
        self._synthesis_queue: asyncio.Queue[_SynthesisRequest] = asyncio.Queue()

        self._playback_service = playback_service
        self._playback_queue: asyncio.Queue[_PlaybackItem] = asyncio.Queue()
        self._delivered_phrases: list[str] = []

        self._shutdown_event = asyncio.Event()
        self._start_event = start_event
        self._events_ready = asyncio.Event()

        self._listening = False
        self._active_reply_id: ReplyId | None = None
        self._reply_active = False
        self._accept_reply_phrases = False
        self._synthesis_active = False
        self._playback_active = False
        self._interrupting = False
        self._barge_in_speech_frames = 0
        self._duck_requested = False
        self._playback_control_queue: asyncio.Queue[_PlaybackControl] = asyncio.Queue()
        self._interaction_started_ns: int | None = None
        self._observed_timing_stages: set[InteractionStage] = set()
        self._playback_started_phrases: set[tuple[ReplyId, PhraseId]] = set()
        self._telemetry_frame_count = 0
        self._telemetry_sample_count = 0
        self._published_vad_detected = False

        self._logger.debug("INITIALIZED")

    async def run(self) -> None:
        async with (
            self._capture_service,
            self._transcription_service,
            self._synthesis_service,
            self._playback_service,
        ):
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(self._events_loop())]
                await self._events_ready.wait()
                self._event_bus.publish(SpeechReady())

                if self._start_event is not None:
                    await self._start_event.wait()
                if self._shutdown_event.is_set():
                    return

                self._logger.debug("Starting tasks")
                self._set_listening(not self._options.wakeword_disabled)
                tasks.extend(
                    [
                        group.create_task(self._capture_loop()),
                        group.create_task(self._segmentation_loop()),
                        group.create_task(self._transcription_loop()),
                        group.create_task(self._synthesis_loop()),
                        group.create_task(self._playback_loop()),
                        group.create_task(self._playback_control_loop()),
                    ]
                )

                await self._shutdown_event.wait()

                self._logger.debug("Canceling tasks")

                for task in tasks:
                    task.cancel()

    async def _capture_loop(self) -> None:
        async for chunk in self._capture_service.capture():
            if self._shutdown_event.is_set():
                return

            self._publish_capture_telemetry(chunk)

            if not self._listening:
                self._observe_pending_turn(chunk)
                self._capture_queue.put_nowait(chunk)
                self._observe_barge_in(chunk)
                continue

            if chunk.is_wakeword:
                self._set_listening(False)
                self._start_interaction()
                self._prepare_for_reply()
                self._event_bus.publish(GenerateReply(ConversationActivated()))

    def _publish_capture_telemetry(self, chunk: SpeechChunk) -> None:
        self._telemetry_frame_count += 1
        self._telemetry_sample_count += len(chunk.audio.samples)
        detection_started = chunk.is_speech and not self._published_vad_detected
        if (
            self._telemetry_frame_count < self._TELEMETRY_INTERVAL_FRAMES
            and not detection_started
            and not chunk.is_wakeword
        ):
            return

        self._event_bus.publish(
            SpeechChunkCaptured(
                samples_len=self._telemetry_sample_count,
                is_speech=chunk.is_speech,
                is_wakeword=chunk.is_wakeword,
            ),
            VADObserved.from_chunk(chunk),
            WakeWordObserved.from_chunk(chunk),
        )
        self._telemetry_frame_count = 0
        self._telemetry_sample_count = 0
        self._published_vad_detected = chunk.is_speech

    async def _events_loop(self) -> None:
        with self._event_bus.subscribe(
            ReplyPhrase,
            ReplyGenerationCompleted,
            ReplyGenerationStarted,
            ShutdownEvent,
        ) as events:
            self._events_ready.set()
            async for event in events:
                try:
                    match event:
                        case ReplyGenerationStarted(reply_id):
                            if not self._interrupting:
                                self._prepare_for_reply(reply_id)
                                self._observe_timing("reply_started")
                        case ReplyPhrase(reply_id, phrase_id, text):
                            if (
                                self._accept_reply_phrases
                                and reply_id == self._active_reply_id
                            ):
                                self._observe_timing("first_reply_phrase")
                                self._synthesis_queue.put_nowait(
                                    _SynthesisRequest(reply_id, phrase_id, text)
                                )
                        case ReplyGenerationCompleted(reply_id):
                            if reply_id == self._active_reply_id:
                                self._reply_active = False
                        case ShutdownEvent():
                            self._shutdown_event.set()
                finally:
                    events.task_done()

    async def _segmentation_loop(self) -> None:
        while not self._shutdown_event.is_set():
            speech_chunk = await self._capture_queue.get()

            try:
                detected, speech_segment = self._utterance_segmenter.feed(speech_chunk)
                if detected and speech_segment:
                    if not self._pending_turn_text:
                        self._start_interaction()
                    self._segmentation_queue.put_nowait(speech_segment)
            finally:
                self._capture_queue.task_done()

    async def _transcription_loop(self) -> None:
        while not self._shutdown_event.is_set():
            speech_segment = await self._segmentation_queue.get()
            try:
                partial_text = ""
                turn_id = self._ensure_turn_id()
                async for item in self._transcription_service.transcribe(
                    speech_segment.audio
                ):
                    match item:
                        case None:
                            self._pending_turn_continuation = False
                            self._pending_turn_silence_frames = 0
                            if not partial_text and not self._pending_turn_text:
                                self._turn_id = None
                        case TranscriptionChunk():
                            partial_text += item.content
                            preview = self._combine_turn_text(partial_text)
                            self._event_bus.publish(
                                TranscriptionProgressObserved(
                                    turn_id=turn_id,
                                    content=preview,
                                    likely_complete=(
                                        self._turn_endpoint_detector.is_complete(
                                            preview
                                        )
                                    ),
                                )
                            )
                        case TranscriptionText():
                            self._observe_timing("transcription_completed")
                            self._handle_transcription(item.content)
            finally:
                self._segmentation_queue.task_done()

    async def _synthesis_loop(self) -> None:
        while not self._shutdown_event.is_set():
            request = await self._synthesis_queue.get()

            try:
                if request.reply_id != self._active_reply_id:
                    continue
                self._synthesis_active = True
                produced_audio = False
                async for chunk in self._synthesis_service.synthesize(request.text):
                    if not self._accept_reply_phrases:
                        break
                    produced_audio = True
                    self._observe_timing("first_audio_synthesized")
                    self._playback_queue.put_nowait(
                        _ReplyAudioFrame(
                            reply_id=request.reply_id,
                            phrase_id=request.phrase_id,
                            frame=chunk,
                        )
                    )
                if produced_audio and self._accept_reply_phrases:
                    # A phrase is delivered only after every preceding audio
                    # frame has completed on the sequential playback queue.
                    self._playback_queue.put_nowait(
                        _ReplyPhraseBoundary(
                            reply_id=request.reply_id,
                            phrase_id=request.phrase_id,
                            text=request.text,
                        )
                    )
            finally:
                self._synthesis_active = False
                self._synthesis_queue.task_done()

    async def _playback_loop(self) -> None:
        while not self._shutdown_event.is_set():
            item = await self._playback_queue.get()

            try:
                if not self._accept_reply_phrases:
                    continue
                match item:
                    case _ReplyAudioFrame(reply_id, phrase_id, frame):
                        if reply_id != self._active_reply_id:
                            continue
                        phrase_key = (reply_id, phrase_id)
                        if phrase_key not in self._playback_started_phrases:
                            self._playback_started_phrases.add(phrase_key)
                            self._event_bus.publish(
                                ReplyPhrasePlaybackStarted(reply_id, phrase_id)
                            )
                        self._playback_active = True
                        self._observe_timing("playback_started")
                        outcome = await self._playback_service.play(frame)
                        if outcome is AudioPlaybackOutcome.INTERRUPTED:
                            self._accept_reply_phrases = False
                    case _ReplyPhraseBoundary(reply_id, phrase_id, text):
                        if reply_id != self._active_reply_id:
                            continue
                        self._delivered_phrases.append(text)
                        self._event_bus.publish(
                            ReplyPhraseDelivered(reply_id, phrase_id)
                        )
            finally:
                self._playback_active = False
                self._playback_queue.task_done()

    async def _playback_control_loop(self) -> None:
        while not self._shutdown_event.is_set():
            control = await self._playback_control_queue.get()
            try:
                match control:
                    case _PlaybackControl.DUCK:
                        await self._playback_service.duck()
                    case _PlaybackControl.RESTORE:
                        await self._playback_service.restore()
                    case _PlaybackControl.INTERRUPT:
                        await self._playback_service.interrupt()
                        await self._playback_service.restore()
                        self._observe_timing("playback_interrupted")
            finally:
                self._playback_control_queue.task_done()

    def _set_listening(self, enabled: bool) -> None:
        self._listening = enabled
        if enabled:
            self._logger.debug("Listening ENABLED")
            self._capture_service.enable_wakeword()
            mode = VoiceSessionMode.WAITING_FOR_WAKE_WORD
        else:
            self._logger.debug("Listening DISABLED")
            self._capture_service.disable_wakeword()
            mode = VoiceSessionMode.ACTIVE
        self._event_bus.publish(VoiceSessionModeChanged(mode))

    def _prepare_for_reply(self, reply_id: ReplyId | None = None) -> None:
        self._active_reply_id = reply_id
        self._reply_active = True
        self._accept_reply_phrases = True
        self._interrupting = False
        self._barge_in_speech_frames = 0
        self._duck_requested = False
        self._delivered_phrases.clear()
        self._playback_started_phrases.clear()

    def _observe_barge_in(self, chunk: SpeechChunk) -> None:
        if not self._assistant_active or self._interrupting:
            self._barge_in_speech_frames = 0
            return

        if not chunk.is_speech:
            self._barge_in_speech_frames = 0
            if self._duck_requested:
                self._duck_requested = False
                self._playback_control_queue.put_nowait(_PlaybackControl.RESTORE)
            return

        if not self._duck_requested:
            self._duck_requested = True
            self._playback_control_queue.put_nowait(_PlaybackControl.DUCK)
        self._barge_in_speech_frames += 1
        if self._barge_in_speech_frames < self._options.barge_in_speech_frames:
            return

        self._interrupting = True
        self._observe_timing("barge_in_detected")
        self._accept_reply_phrases = False
        self._reply_active = False
        self._synthesis_service.interrupt()
        self._drain(self._synthesis_queue)
        self._drain(self._playback_queue)
        self._event_bus.publish(
            CancelReply(
                spoken_text=" ".join(self._delivered_phrases),
                reply_id=self._active_reply_id,
            )
        )
        self._duck_requested = False
        self._playback_control_queue.put_nowait(_PlaybackControl.INTERRUPT)
        self._logger.debug("Reply INTERRUPTED")

    def _observe_pending_turn(self, chunk: SpeechChunk) -> None:
        if not self._pending_turn_text:
            return
        if chunk.is_speech:
            self._pending_turn_continuation = True
            self._pending_turn_silence_frames = 0
            return
        if self._pending_turn_continuation:
            return
        self._pending_turn_silence_frames += 1
        if (
            self._pending_turn_silence_frames
            >= self._options.continuation_silence_frames
        ):
            self._commit_user_turn(self._pending_turn_text)

    def _handle_transcription(self, text: str) -> None:
        combined = self._combine_turn_text(text)
        self._pending_turn_continuation = False
        self._pending_turn_silence_frames = 0
        if not combined:
            self._turn_id = None
            return
        if not self._turn_endpoint_detector.is_complete(combined):
            self._pending_turn_text = combined
            return
        self._commit_user_turn(combined)

    def _commit_user_turn(self, text: str) -> None:
        turn_id = self._ensure_turn_id()
        self._pending_turn_text = ""
        self._pending_turn_continuation = False
        self._pending_turn_silence_frames = 0
        self._turn_id = None
        self._prepare_for_reply()
        self._event_bus.publish(
            UserTurnCommitted(turn_id=turn_id, text=text),
            GenerateReply(UserTurn(text)),
        )

    def _ensure_turn_id(self) -> TurnId:
        if self._turn_id is None:
            self._turn_sequence += 1
            self._turn_id = self._turn_sequence
        return self._turn_id

    def _combine_turn_text(self, text: str) -> str:
        return " ".join(
            part for part in (self._pending_turn_text, text.strip()) if part
        )

    @property
    def _assistant_active(self) -> bool:
        return (
            self._reply_active
            or self._synthesis_active
            or self._playback_active
            or not self._synthesis_queue.empty()
            or not self._playback_queue.empty()
        )

    def _start_interaction(self) -> None:
        self._interaction_started_ns = perf_counter_ns()
        self._observed_timing_stages.clear()
        self._observe_timing("turn_ready")

    def _observe_timing(self, stage: InteractionStage) -> None:
        started_ns = self._interaction_started_ns
        if started_ns is None or stage in self._observed_timing_stages:
            return
        self._observed_timing_stages.add(stage)
        self._event_bus.publish(
            InteractionTimingObserved(
                stage=stage,
                elapsed_ms=(perf_counter_ns() - started_ns) / 1_000_000,
            )
        )

    @staticmethod
    def _drain(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                queue.task_done()
