import asyncio

from mlx_audio.tts.models.moss_tts.processor import UserMessage

from henry_common.components import Component
from henry_common.events import EventBus, ShutdownEvent

from .events import GenerateReply, ReplyCompleted, ReplyLine, ReplyStarted
from .graph import ReplyGraph


class Worker(Component):
    def __init__(self, event_bus: EventBus, graph: ReplyGraph) -> None:
        super().__init__()
        self._event_bus = event_bus

        self._graph = graph
        self._graph_queue: asyncio.Queue[str] = asyncio.Queue()

        self._shutdown_event = asyncio.Event()

        self._logger.debug("INITIALIZED")

    async def run(self) -> None:
        self._logger.debug("Starting tasks")

        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self._events_loop()),
                group.create_task(self._graph_loop()),
            ]

            await self._shutdown_event.wait()

            self._logger.debug("Canceling tasks")

            for task in tasks:
                task.cancel()

    async def _events_loop(self) -> None:
        with self._event_bus.subscribe(
            GenerateReply,
            ShutdownEvent,
        ) as events:
            async for event in events:
                match event:
                    case GenerateReply(text):
                        if text is None:
                            self._event_bus.publish(
                                ReplyStarted(),
                                ReplyLine(text="Dzień dobry!"),
                                ReplyCompleted(),
                            )
                        else:
                            self._logger.debug("Generating reply")
                            self._graph_queue.put_nowait(text)
                    case ShutdownEvent():
                        self._shutdown_event.set()
                events.task_done()

    async def _graph_loop(self) -> None:
        while not self._shutdown_event.is_set():
            text = await self._graph_queue.get()

            try:
                async for chunk in self._graph.compiled.astream_events(
                    input=UserMessage(text)
                ):
                    self._logger.debug(chunk)

                self._event_bus.publish(
                    ReplyStarted(),
                    ReplyLine(text="Dzień dobry!"),
                    ReplyCompleted(),
                )
            finally:
                self._graph_queue.task_done()
