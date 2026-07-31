import asyncio
from dataclasses import dataclass

from loguru import logger

from henry_speech.events import (
    AppEvent,
    AppEventSink,
    AudioCaptured,
    AudioPlayed,
    TelemetryEvent,
)


class EventLogger(AppEventSink):
    """Write application events to the configured Loguru sink."""

    def __init__(self) -> None:
        self._logger = logger.bind(component="EventTracker")

    def publish(self, *events: AppEvent) -> None:
        for event in events:
            self._logger.trace("{}", event, event=event.__class__.__name__)


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    captured_sample_count: int = 0
    played_sample_count: int = 0
    speech_score: float = 0.0
    speech_detected: bool = False
    wakeword_score: float = 0.0
    wakeword_detected: bool = False


class EventBridge(AppEventSink):
    def __init__(self):
        self._captured_sample_count = 0
        self._played_sample_count = 0
        self._speech_score = 0.0
        self._speech_detected = False
        self._wakeword_score = 0.0
        self._wakeword_detected = False
        self._queue: asyncio.Queue[AppEvent] = asyncio.Queue()

    @property
    def telemetry_snapshot(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            captured_sample_count=self._captured_sample_count,
            played_sample_count=self._played_sample_count,
            speech_score=self._speech_score,
            speech_detected=self._speech_detected,
            wakeword_score=self._wakeword_score,
            wakeword_detected=self._wakeword_detected,
        )

    def publish(self, *events: AppEvent) -> None:
        for event in events:
            match event:
                case AudioCaptured():
                    self._speech_score = event.speech_score
                    self._speech_detected = event.speech_detected
                    self._captured_sample_count += event.samples_count
                    if (
                        event.wakeword_detected is not None
                        and event.wakeword_score is not None
                    ):
                        self._wakeword_score = event.wakeword_score
                        self._wakeword_detected = event.wakeword_detected
                    else:
                        self._wakeword_score = 0.0
                        self._wakeword_detected = False

                case AudioPlayed():
                    self._played_sample_count += event.samples_count
                case TelemetryEvent():
                    pass
                case _:
                    self._queue.put_nowait(event)

    async def receive(self) -> AppEvent:
        """Wait for the next non-telemetry application event."""
        return await self._queue.get()
