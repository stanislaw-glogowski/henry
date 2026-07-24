import asyncio
import queue
import threading
from collections import deque
from collections.abc import AsyncIterator, Iterator

import numpy as np

from henry_client.audio import AudioChunk, AudioFormat, AudioFrame
from henry_client.events import AppEvent, AppEventSink
from henry_client.reply import (
    ReplyChunk,
    ReplyLine,
    ReplyRequest,
    ReplySignal,
    ReplyText,
)


def frame(
    value: float = 0.0,
    *,
    samples_count: int = 4,
    sample_rate: int = 16_000,
) -> AudioFrame:
    return AudioFrame(
        format=AudioFormat(sample_rate=sample_rate),
        samples=np.full(samples_count, value, dtype=np.float32),
    )


def chunk(
    *,
    sequence_id: int = 1,
    vad_score: float = 0.0,
    wakeword_score: float | None = None,
) -> AudioChunk:
    return AudioChunk(
        sequence_id=sequence_id,
        frame=frame(),
        vad_score=vad_score,
        wakeword_score=wakeword_score,
    )


class FakeInputStream:
    def __init__(self, open_error: BaseException | None = None) -> None:
        self.items: queue.Queue[AudioFrame | BaseException] = queue.Queue()
        self.fallback = frame()
        self.open_error = open_error
        self.opened = False

    def open(self) -> None:
        if self.open_error is not None:
            raise self.open_error
        self.opened = True

    def close(self) -> None:
        self.opened = False

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
    def __init__(
        self,
        error: BaseException | None = None,
        *,
        open_error: BaseException | None = None,
    ) -> None:
        self.frames: list[AudioFrame] = []
        self.error = error
        self.open_error = open_error
        self.opened = False

    def open(self) -> None:
        if self.open_error is not None:
            raise self.open_error
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def write(self, value: AudioFrame) -> None:
        if self.error is not None:
            raise self.error
        self.frames.append(value)


class FakeVADModel:
    def __init__(self, *scores: float) -> None:
        self.scores = deque(scores)
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def predict(self, value: AudioFrame) -> float:
        return self.scores.popleft() if self.scores else 0.0


class FakeWakeWordModel:
    def __init__(self, *scores: float) -> None:
        self.scores = deque(scores)
        self.reset_threads: list[int] = []
        self.reset_event = threading.Event()
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def predict(self, value: AudioFrame) -> float:
        return self.scores.popleft() if self.scores else 0.0

    def reset(self) -> None:
        self.reset_threads.append(threading.get_ident())
        self.reset_event.set()


class FakeSTTModel:
    def __init__(
        self,
        text: str | None = "transcript",
        *,
        open_error: BaseException | None = None,
        transcription_error: BaseException | None = None,
    ) -> None:
        self.text = text
        self.open_error = open_error
        self.transcription_error = transcription_error
        self.opened = False
        self.thread_ids: list[int] = []

    def open(self) -> None:
        self.thread_ids.append(threading.get_ident())
        if self.open_error is not None:
            raise self.open_error
        self.opened = True

    def close(self) -> None:
        self.thread_ids.append(threading.get_ident())
        self.opened = False

    def transcribe(self, value: AudioFrame) -> str | None:
        self.thread_ids.append(threading.get_ident())
        if self.transcription_error is not None:
            raise self.transcription_error
        return self.text


class FakeTTSModel:
    def __init__(
        self,
        *,
        open_error: BaseException | None = None,
        synthesis_error: BaseException | None = None,
    ) -> None:
        self.texts: list[str] = []
        self.open_error = open_error
        self.synthesis_error = synthesis_error
        self.opened = False
        self.thread_ids: list[int] = []

    def open(self) -> None:
        self.thread_ids.append(threading.get_ident())
        if self.open_error is not None:
            raise self.open_error
        self.opened = True

    def close(self) -> None:
        self.thread_ids.append(threading.get_ident())
        self.opened = False

    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        self.thread_ids.append(threading.get_ident())
        if self.synthesis_error is not None:
            raise self.synthesis_error
        self.texts.append(text)
        yield frame(value=float(len(self.texts)), sample_rate=22_050)


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
        self.wakeword_enabled = False
        self.wakeword_resets = 0

    async def capture(self) -> AsyncIterator[AudioChunk]:
        while True:
            value = await self.chunks.get()
            if value is None:
                return
            yield value

    async def playback(self, value: AudioFrame) -> None:
        self.written.append(value)

    def enable_wakeword(self) -> None:
        self.wakeword_enabled = True

    def disable_wakeword(self) -> None:
        self.wakeword_enabled = False

    def reset_wakeword(self) -> None:
        self.wakeword_resets += 1


class FakeSpeechService:
    def __init__(self, transcript: str | None = "question") -> None:
        self.transcript = transcript
        self.synthesized: list[str] = []

    def segment(
        self,
        value: AudioFrame,
        speech_detected: bool,
    ) -> tuple[bool, AudioFrame | None]:
        return True, frame(value=0.5)

    async def transcribe(self, value: AudioFrame) -> str | None:
        return self.transcript

    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        self.synthesized.append(text)
        yield frame(value=float(len(self.synthesized)), sample_rate=22_050)


class FakeReplyService:
    def __init__(
        self,
        *lines: str,
        activation_text: str | None = None,
    ) -> None:
        self.lines = lines
        self.activation_text = activation_text
        self.received: list[ReplyRequest] = []

    async def reply(self, request: ReplyRequest):
        self.received.append(request)
        if request is ReplySignal.ACTIVATION:
            if self.activation_text is not None:
                yield ReplyChunk(self.activation_text)
                yield ReplyLine(self.activation_text)
                yield ReplyText(self.activation_text)
            else:
                yield ReplyText("")
            return

        for line in self.lines:
            yield ReplyChunk(line)
            yield ReplyLine(line)
        yield ReplyText("\n".join(self.lines))
