import asyncio
import queue
import threading
from collections import deque
from collections.abc import AsyncIterator, Iterator, Sequence
from types import TracebackType
from typing import Self

import numpy as np

from henry_client.audio import AudioChunk, AudioFrame
from henry_client.conversation.domain import Message, MessageChunk
from henry_client.events import AppEvent, AppEventSink


def frame(
    value: float = 0.0,
    *,
    samples_count: int = 4,
    sample_rate: int = 16_000,
) -> AudioFrame:
    return AudioFrame(
        samples=np.full(samples_count, value, dtype=np.float32),
        sample_rate=sample_rate,
        channels=1,
    )


def chunk(
    *,
    speech_detected: bool = False,
    speech_score: float = 0.0,
    wakeword_detected: bool | None = None,
    wakeword_score: float | None = None,
) -> AudioChunk:
    source = frame()
    return source.build_chunk(
        speech_detected=speech_detected,
        speech_score=speech_score,
        wakeword_detected=wakeword_detected,
        wakeword_score=wakeword_score,
    )


class FakeInputStream:
    def __init__(self) -> None:
        self.items: queue.Queue[AudioFrame | BaseException] = queue.Queue()
        self.fallback = frame()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def feed(self, *items: AudioFrame | BaseException) -> None:
        for item in items:
            self.items.put_nowait(item)

    def read(self) -> AudioFrame:
        try:
            item = self.items.get(timeout=0.01)
        except queue.Empty:
            return self.fallback
        if isinstance(item, BaseException):
            raise item
        return item


class FakeOutputStream:
    def __init__(self, error: BaseException | None = None) -> None:
        self.frames: list[AudioFrame] = []
        self.error = error

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def write(self, value: AudioFrame) -> None:
        if self.error is not None:
            raise self.error
        self.frames.append(value)


class FakeVADModel:
    def __init__(self, *scores: float) -> None:
        self.scores = deque(scores)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def predict(self, value: AudioFrame) -> float:
        return self.scores.popleft() if self.scores else 0.0


class FakeWakeWordModel:
    def __init__(self, *scores: float) -> None:
        self.scores = deque(scores)
        self.reset_threads: list[int] = []
        self.reset_event = threading.Event()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def predict(self, value: AudioFrame) -> float:
        return self.scores.popleft() if self.scores else 0.0

    def reset(self) -> None:
        self.reset_threads.append(threading.get_ident())
        self.reset_event.set()


class FakeSTTModel:
    def __init__(self, text: str | None = "transcript") -> None:
        self.text = text

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def transcribe(self, value: AudioFrame) -> str | None:
        return self.text


class FakeTTSModel:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        self.texts.append(text)
        yield frame(value=float(len(self.texts)), sample_rate=22_050)


class FakeLanguageModel:
    def __init__(self, *parts: str) -> None:
        self.parts = parts
        self.messages: list[Sequence[Message]] = []

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def generate(self, messages: Sequence[Message]) -> Iterator[MessageChunk]:
        self.messages.append(messages)
        for part in self.parts:
            yield MessageChunk(part)


class RecordingEventSink(AppEventSink):
    def __init__(self) -> None:
        self.events: list[AppEvent] = []
        self.changed = asyncio.Event()

    def publish(self, *events: AppEvent) -> None:
        self.events.extend(events)
        self.changed.set()


class FakeAudioService:
    def __init__(self) -> None:
        self.chunks: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self.written: list[AudioFrame] = []

    async def read(self) -> AsyncIterator[AudioChunk]:
        while True:
            value = await self.chunks.get()
            if value is None:
                return
            yield value

    async def write(self, value: AudioFrame) -> None:
        self.written.append(value)


class FakeSpeechService:
    def __init__(self, transcript: str | None = "question") -> None:
        self.transcript = transcript
        self.synthesized: list[str] = []

    def detect(self, value: AudioChunk) -> tuple[bool, AudioFrame | None]:
        return True, frame(value=0.5)

    async def transcribe(self, value: AudioFrame) -> str | None:
        return self.transcript

    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        self.synthesized.append(text)
        yield frame(value=float(len(self.synthesized)), sample_rate=22_050)


class FakeConversationService:
    def __init__(self, *lines: str) -> None:
        self.lines = lines
        self.received: list[str] = []

    async def generate_reply(self, text: str):
        from henry_client.conversation import MessageLine

        self.received.append(text)
        for line in self.lines:
            yield MessageLine(line)
        yield "\n".join(self.lines)
