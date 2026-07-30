from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Self


@dataclass(frozen=True, slots=True)
class Event: ...


class StateEvent(Event): ...


class TelemetryEvent(Event): ...


class ShutdownEvent(Event): ...


type EventQueue = asyncio.Queue[Event | None]
type EventTypes = tuple[type[Event], ...]


class EventSubscription(
    AsyncIterator[Event],
    AbstractContextManager,
):
    def __init__(
        self,
        bus: EventBus,
        queue: EventQueue,
    ) -> None:
        self._bus = bus
        self._queue = queue
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        self.close()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> Event:
        if self._closed:
            raise StopAsyncIteration

        event = await self._queue.get()

        if event is None:
            self._closed = True
            raise StopAsyncIteration

        return event

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._bus.unsubscribe(self._queue)


class EventBus(AbstractContextManager):
    _QUEUE_MAXSIZE = 1000

    def __init__(self) -> None:
        self._subscriptions: dict[EventQueue, EventTypes] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        self.close()

    def publish(self, *events: Event) -> None:
        for queue, event_types in self._subscriptions.items():
            for event in events:
                if not event_types or isinstance(event, event_types):
                    queue.put_nowait(event)

    def subscribe(
        self,
        *event_types: type[Event],
    ) -> EventSubscription:
        queue: EventQueue = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._subscriptions[queue] = event_types

        return EventSubscription(self, queue)

    def unsubscribe(self, queue: EventQueue) -> None:
        if self._subscriptions.pop(queue, None) is None:
            return

        queue.put_nowait(None)

    def close(self) -> None:
        for queue in self._subscriptions:
            queue.put_nowait(None)
        self._subscriptions.clear()
