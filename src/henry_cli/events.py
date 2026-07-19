import asyncio

from henry import AppEventSink
from henry.events import AppEvent, AppEventBatch


class EventBridge(AppEventSink):
    def __init__(self):
        self._queue: asyncio.Queue[AppEventBatch] = asyncio.Queue()

    def publish(self, *events: AppEvent) -> None:
        self._queue.put_nowait(events)

    async def receive(self) -> AppEventBatch:
        return await self._queue.get()
