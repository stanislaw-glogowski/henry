import asyncio
from contextlib import suppress

from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from henry_common.components import Component
from henry_common.events import EventBus, ShutdownEvent

from .domain import ConversationTextChunk
from .events import (
    CancelReply,
    ConversationActivated,
    ConversationInput,
    GenerateReply,
    ReplyChunk,
    ReplyGenerationCompleted,
    ReplyGenerationStarted,
    ReplyPhrase,
    UserTurn,
)
from .graph import ConversationContext, ConversationGraph
from .preparation import ProfilePreparation
from .reply import ReplySegmenter


class Worker(Component):
    THREAD_ID = "default"

    def __init__(
        self,
        event_bus: EventBus,
        graph: ConversationGraph,
        context: ConversationContext,
        profile_preparation: ProfilePreparation | None = None,
    ) -> None:
        super().__init__()
        self._event_bus = event_bus
        self._graph = graph
        self._context = context
        self._profile_preparation = profile_preparation
        self._graph_queue: asyncio.Queue[ConversationInput] = asyncio.Queue()
        self._active_reply: asyncio.Task[None] | None = None
        self._delivery_context = ""
        self._shutdown_event = asyncio.Event()
        self._logger.debug("INITIALIZED")

    async def run(self) -> None:
        self._logger.debug("Starting tasks")

        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(self._events_loop()),
                group.create_task(self._graph_loop()),
            ]
            if self._profile_preparation is not None:
                tasks.append(group.create_task(self._prepare_profile()))

            await self._shutdown_event.wait()
            self._logger.debug("Canceling tasks")

            for task in tasks:
                task.cancel()

    async def _prepare_profile(self) -> None:
        preparation = self._profile_preparation
        if preparation is None:
            return
        try:
            await preparation.prepare()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._logger.warning("Profile preparation FAILED: {}", error)

    async def _events_loop(self) -> None:
        with self._event_bus.subscribe(
            CancelReply,
            GenerateReply,
            ShutdownEvent,
        ) as events:
            async for event in events:
                try:
                    match event:
                        case GenerateReply(input):
                            self._graph_queue.put_nowait(input)
                        case CancelReply(spoken_text):
                            await self._cancel_reply(spoken_text)
                        case ShutdownEvent():
                            self._shutdown_event.set()
                finally:
                    events.task_done()

    async def _graph_loop(self) -> None:
        while not self._shutdown_event.is_set():
            conversation_input = await self._graph_queue.get()

            try:
                self._event_bus.publish(ReplyGenerationStarted())
                self._active_reply = asyncio.create_task(
                    self._stream(conversation_input)
                )
                try:
                    await self._active_reply
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
            finally:
                self._active_reply = None
                self._event_bus.publish(ReplyGenerationCompleted())
                self._graph_queue.task_done()

    async def _cancel_reply(self, spoken_text: str) -> None:
        active_reply = self._active_reply
        if active_reply is not None:
            active_reply.cancel()
            with suppress(asyncio.CancelledError):
                await active_reply

        while True:
            try:
                self._graph_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._graph_queue.task_done()

        self._delivery_context = (
            # The next graph run receives delivery state as input instead of
            # mutating a checkpoint while its current run is being cancelled.
            "The previous answer was interrupted. The user heard only this "
            f"prefix: {spoken_text!r}. Do not assume they heard the remainder."
            if spoken_text
            else "The previous answer was interrupted before any part was delivered. "
            "Do not assume the user heard it."
        )

    async def _stream(self, conversation_input: ConversationInput) -> None:
        delivery_context, self._delivery_context = self._delivery_context, ""
        match conversation_input:
            case ConversationActivated():
                graph_input = {
                    "delivery_context": delivery_context,
                    "input_kind": "activation",
                    "messages": [],
                }
            case UserTurn(text):
                graph_input = {
                    "delivery_context": delivery_context,
                    "input_kind": "user_turn",
                    "messages": [HumanMessage(content=text)],
                }

        config: RunnableConfig = {
            "configurable": {"thread_id": self.THREAD_ID},
        }
        segmenter = ReplySegmenter()

        async for event in self._graph.compiled.astream(
            input=graph_input,
            config=config,
            context=self._context,
            stream_mode="custom",
        ):
            if not isinstance(event, ConversationTextChunk):
                continue

            chunk = event.content
            if not chunk:
                continue

            self._event_bus.publish(ReplyChunk(text=chunk))
            for phrase in segmenter.feed(chunk):
                self._event_bus.publish(ReplyPhrase(text=phrase))

        for phrase in segmenter.flush():
            self._event_bus.publish(ReplyPhrase(text=phrase))
