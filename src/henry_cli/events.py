import asyncio
from dataclasses import dataclass

from henry_client.events import (
    AppEvent,
    AppEventSink,
    AudioCaptured,
    AudioPlayed,
    TelemetryEvent,
)


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    vad_score: float = 0.0
    is_speech: bool = False
    captured_sample_count: int = 0
    played_sample_count: int = 0


class EventBridge(AppEventSink):
    def __init__(self):
        self._vad_score = 0.0
        self._is_speech = False
        self._captured_sample_count = 0
        self._played_sample_count = 0
        self._queue: asyncio.Queue[AppEvent] = asyncio.Queue()

    @property
    def telemetry_snapshot(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            vad_score=self._vad_score,
            is_speech=self._is_speech,
            captured_sample_count=self._captured_sample_count,
            played_sample_count=self._played_sample_count,
        )

    def publish(self, *events: AppEvent) -> None:
        for event in events:
            match event:
                case AudioCaptured():
                    self._vad_score = event.vad_score
                    self._is_speech = event.is_speech
                    self._captured_sample_count += event.samples_count
                case AudioPlayed():
                    self._played_sample_count += event.samples_count
                case TelemetryEvent():
                    pass
                case _:
                    self._queue.put_nowait(event)

    async def receive(self) -> AppEvent:  # noqa: F821
        return await self._queue.get()
