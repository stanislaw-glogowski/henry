import asyncio
from dataclasses import dataclass

from ..audio import AudioChunk, AudioFrame
from ..concurrency import put_latest
from ..events import AppEventSink
from ..lifecycle import AsyncManagedResource
from .domain import VAD
from .events import VADObserved


@dataclass
class SpeechConfig:
    vad_threshold: float = 0.5


class SpeechService(AsyncManagedResource):
    def __init__(
        self,
        vad: VAD,
        events: AppEventSink,
        config: SpeechConfig | None = None,
    ) -> None:
        if config is None:
            config = SpeechConfig()

        self._vad = vad
        self._events = events
        self._config = config

    async def run(
        self,
        frames: asyncio.Queue[AudioFrame],
        chunks: asyncio.Queue[AudioChunk],
    ) -> None:
        while True:
            frame = await frames.get()

            try:
                vad_score = self._vad.predict(frame)
                is_speech = vad_score > self._config.vad_threshold

                self._events.publish(
                    VADObserved(
                        score=vad_score,
                        is_speech=is_speech,
                    )
                )

                put_latest(
                    chunks,
                    frame.build_chunk(is_speech),
                )
            finally:
                frames.task_done()

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass
