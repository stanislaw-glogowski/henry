import asyncio

from langchain.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig

from henry_common.components import Component
from henry_common.events import EventBus, ShutdownEvent

from .events import (
    GenerateReply,
    ReplyChunk,
    ReplyCompleted,
    ReplyLine,
    ReplyStarted,
)
from .graph import ReplyContext, ReplyGraph, ReplyNode


class Worker(Component):
    THREAD_ID = "default"

    def __init__(
        self,
        event_bus: EventBus,
        graph: ReplyGraph,
        context: ReplyContext,
    ) -> None:
        super().__init__()
        self._event_bus = event_bus

        self._graph = graph
        self._context = context
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
                self._event_bus.publish(ReplyStarted())

                config: RunnableConfig = {
                    "configurable": {"thread_id": self.THREAD_ID},
                }

                line_buffer = ""
                async for message, metadata in self._graph.compiled.astream(
                    input={"messages": [HumanMessage(content=text)]},
                    config=config,
                    context=self._context,
                    stream_mode="messages",
                ):
                    if metadata.get("langgraph_node") != ReplyNode.NAME:
                        continue
                    if not isinstance(message, AIMessageChunk):
                        continue

                    chunk = message.text
                    if not chunk:
                        continue

                    self._event_bus.publish(ReplyChunk(text=chunk))
                    line_buffer += chunk

                    while "\n" in line_buffer:
                        line, line_buffer = line_buffer.split("\n", maxsplit=1)
                        if line:
                            self._event_bus.publish(ReplyLine(text=line))

                if line_buffer:
                    self._event_bus.publish(ReplyLine(text=line_buffer))
            finally:
                self._event_bus.publish(ReplyCompleted())
                self._graph_queue.task_done()
