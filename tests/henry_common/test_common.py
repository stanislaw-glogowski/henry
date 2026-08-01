import asyncio
import threading
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from henry_common.components import AbstractAsyncService, AbstractResource
from henry_common.events import Event, EventBus
from henry_common.logger import bind_logger
from henry_common.validation import ConfigModel


class Resource(AbstractResource):
    def __init__(self) -> None:
        super().__init__("test")
        self.calls: list[str] = []

    def open(self) -> None:
        self.calls.append("open")

    def close(self) -> None:
        self.calls.append("close")


class Service(AbstractAsyncService):
    def __init__(
        self,
        *,
        fail_open: bool = False,
        fail_close: bool = False,
        fail_post_stop: bool = False,
    ) -> None:
        super().__init__()
        self.fail_open = fail_open
        self.fail_close = fail_close
        self.fail_post_stop = fail_post_stop
        self.calls: list[str] = []
        self.thread_ids: set[int] = set()

    def _open_resources(self) -> None:
        self.calls.append("open")
        self.thread_ids.add(threading.get_ident())
        if self.fail_open:
            raise RuntimeError("open")

    def _close_resources(self) -> None:
        self.calls.append("close")
        self.thread_ids.add(threading.get_ident())
        if self.fail_close:
            raise RuntimeError("close")

    async def _post_stop(self) -> None:
        self.calls.append("post_stop")
        if self.fail_post_stop:
            raise RuntimeError("post_stop")

    async def execute(self) -> int:
        return await self._run_in_executor(threading.get_ident)


@dataclass(frozen=True, slots=True)
class First(Event):
    value: int


@dataclass(frozen=True, slots=True)
class Second(Event):
    value: int


def test_resource_context_and_bound_logger() -> None:
    resource = Resource()
    with resource as entered:
        assert entered is resource
    assert resource.calls == ["open", "close"]

    logger = bind_logger("worker", "default")
    assert logger._options[8]["component"] == "worker(default)"


def test_async_service_lifecycle_and_executor_ownership() -> None:
    async def scenario() -> None:
        service = Service()
        with pytest.raises(RuntimeError, match="is not started"):
            service._require_executor()

        async with service as entered:
            assert entered is service
            worker_thread = await service.execute()
            assert worker_thread != threading.get_ident()
            with pytest.raises(RuntimeError, match="already started"):
                await service.start()

        await service.stop()
        assert service.calls == ["open", "close", "post_stop"]
        assert service.thread_ids == {worker_thread}

    asyncio.run(scenario())


def test_async_service_cleans_up_start_and_stop_errors() -> None:
    async def scenario() -> None:
        failed_start = Service(fail_open=True)
        with pytest.raises(RuntimeError, match="open"):
            await failed_start.start()
        assert failed_start.calls == ["open", "close", "post_stop"]

        failed_stop = Service(fail_close=True, fail_post_stop=True)
        await failed_stop.start()
        await failed_stop.stop()
        assert failed_stop.calls == ["open", "close", "post_stop"]

    asyncio.run(scenario())


def test_event_bus_filters_closes_and_unsubscribes() -> None:
    async def scenario() -> None:
        bus = EventBus()
        filtered = bus.subscribe(First)
        all_events = bus.subscribe()

        assert filtered.__aiter__() is filtered
        bus.publish(First(1), None, Second(2))

        assert await filtered.__anext__() == First(1)
        filtered.task_done()
        assert await all_events.__anext__() == First(1)
        all_events.task_done()
        assert await all_events.__anext__() == Second(2)
        all_events.task_done()

        filtered.close()
        filtered.close()
        with pytest.raises(StopAsyncIteration):
            await filtered.__anext__()

        bus.unsubscribe(asyncio.Queue())
        bus.close()
        with pytest.raises(StopAsyncIteration):
            await all_events.__anext__()

    asyncio.run(scenario())


def test_event_bus_context_closes_subscription() -> None:
    async def scenario() -> None:
        with EventBus() as bus:
            with bus.subscribe() as subscription:
                bus.publish(First(3))
                assert await subscription.__anext__() == First(3)
                subscription.task_done()
        with pytest.raises(StopAsyncIteration):
            await subscription.__anext__()

    asyncio.run(scenario())


def test_config_model_is_frozen_and_forbids_extra_fields() -> None:
    class Example(ConfigModel):
        value: int

    model = Example(value=1)
    with pytest.raises(ValidationError):
        model.value = 2
    with pytest.raises(ValidationError):
        Example(value=1, extra=True)
