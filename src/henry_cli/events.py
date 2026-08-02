import asyncio
from dataclasses import dataclass

from loguru import logger

from henry_common.events import Event, EventBus, ShutdownEvent
from henry_conversation.events import (
    CancelReply,
    ConversationActivated,
    ConversationReady,
    GenerateReply,
    ReplyDraftUpdated,
    ReplyGenerationCompleted,
    ReplyGenerationStarted,
    ReplyPhrase,
    UserTurn,
)
from henry_speech.events import (
    AudioDevicesSelected,
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
    VoiceSessionModeChanged,
    WakeWordObserved,
)


@dataclass(frozen=True, slots=True)
class TranscriptionDraft:
    turn_id: TurnId
    text: str
    likely_complete: bool


@dataclass(frozen=True, slots=True)
class ReplyDraft:
    reply_id: int
    text: str


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    captured_sample_count: int = 0
    vad_score: float = 0.0
    vad_detected: bool = False
    wakeword_score: float = 0.0
    wakeword_detected: bool = False
    transcription: TranscriptionDraft | None = None
    reply: ReplyDraft | None = None
    timings: tuple[tuple[InteractionStage, float], ...] = ()


class UiEventBridge:
    _QUEUE_MAXSIZE = 1_000

    def __init__(self) -> None:
        self._captured_sample_count = 0
        self._vad_score = 0.0
        self._vad_detected = False
        self._wakeword_score = 0.0
        self._wakeword_detected = False
        self._transcription: TranscriptionDraft | None = None
        self._reply: ReplyDraft | None = None
        self._timings: dict[InteractionStage, float] = {}
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._ready = asyncio.Event()

    @property
    def telemetry_snapshot(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            captured_sample_count=self._captured_sample_count,
            vad_score=self._vad_score,
            vad_detected=self._vad_detected,
            wakeword_score=self._wakeword_score,
            wakeword_detected=self._wakeword_detected,
            transcription=self._transcription,
            reply=self._reply,
            timings=tuple(self._timings.items()),
        )

    async def wait_ready(self) -> None:
        await self._ready.wait()

    async def receive(self) -> Event:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    async def run(self, event_bus: EventBus) -> None:
        with event_bus.subscribe(
            AudioDevicesSelected,
            CancelReply,
            ConversationReady,
            InteractionTimingObserved,
            ReplyDraftUpdated,
            ReplyGenerationCompleted,
            ReplyGenerationStarted,
            ReplyPhrase,
            ReplyPhraseDelivered,
            ReplyPhrasePlaybackStarted,
            ShutdownEvent,
            SpeechChunkCaptured,
            SpeechReady,
            TranscriptionProgressObserved,
            UserTurnCommitted,
            VADObserved,
            VoiceSessionModeChanged,
            WakeWordObserved,
        ) as events:
            self._ready.set()
            async for event in events:
                try:
                    if isinstance(event, CancelReply) and self._reply is not None:
                        if (
                            event.reply_id is None
                            or event.reply_id == self._reply.reply_id
                        ):
                            self._queue.put_nowait(
                                ReplyDraftUpdated(
                                    reply_id=self._reply.reply_id,
                                    text=self._reply.text,
                                )
                            )
                    if self._reduce_telemetry(event):
                        continue
                    self._queue.put_nowait(event)
                    if isinstance(event, ShutdownEvent):
                        return
                finally:
                    events.task_done()

    def _reduce_telemetry(self, event: Event) -> bool:
        match event:
            case SpeechChunkCaptured(samples_len):
                self._captured_sample_count += samples_len
            case VADObserved(score, detected):
                self._vad_score = score
                self._vad_detected = detected
            case WakeWordObserved(score, detected):
                self._wakeword_score = score
                self._wakeword_detected = detected
            case TranscriptionProgressObserved(turn_id, content, likely_complete):
                self._transcription = TranscriptionDraft(
                    turn_id=turn_id,
                    text=content,
                    likely_complete=likely_complete,
                )
            case UserTurnCommitted():
                self._transcription = None
                return False
            case ReplyDraftUpdated(reply_id, text):
                self._reply = ReplyDraft(reply_id, text) if text else None
            case ReplyGenerationCompleted(reply_id):
                if self._reply is None or self._reply.reply_id == reply_id:
                    self._reply = None
                return False
            case CancelReply(_, reply_id):
                if (
                    self._reply is None
                    or reply_id is None
                    or self._reply.reply_id == reply_id
                ):
                    self._reply = None
                return False
            case InteractionTimingObserved(stage, elapsed_ms):
                self._timings[stage] = elapsed_ms
            case _:
                return False
        return True


async def run_event_logger(event_bus: EventBus) -> None:
    with event_bus.subscribe(
        GenerateReply,
        ReplyGenerationStarted,
        ReplyPhrase,
        ReplyGenerationCompleted,
        InteractionTimingObserved,
        WakeWordObserved,
        ShutdownEvent,
    ) as events:
        async for event in events:
            try:
                match event:
                    case GenerateReply(ConversationActivated()):
                        logger.info("Wake word activated the conversation")
                    case GenerateReply(UserTurn(text)):
                        logger.info("User: {}", text)
                    case ReplyGenerationStarted(reply_id):
                        logger.debug("Generating response: reply_id={}", reply_id)
                    case ReplyPhrase(reply_id, phrase_id, text):
                        logger.info(
                            "Henry: {}",
                            text,
                            reply_id=reply_id,
                            phrase_id=phrase_id,
                        )
                    case ReplyGenerationCompleted(reply_id):
                        logger.debug("Response completed: reply_id={}", reply_id)
                    case InteractionTimingObserved(stage, elapsed_ms):
                        logger.debug(
                            "Interaction timing: stage='{}', elapsed_ms={:.1f}",
                            stage,
                            elapsed_ms,
                        )
                    case WakeWordObserved(detected=True):
                        logger.debug("Wake word detected")
                    case ShutdownEvent():
                        logger.debug("Shutdown requested")
                        return
            finally:
                events.task_done()
