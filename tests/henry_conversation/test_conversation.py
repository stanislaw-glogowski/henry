import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain.messages import AIMessageChunk, HumanMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

import henry_conversation.graph.nodes as nodes_module
from henry_common.events import EventBus, ShutdownEvent
from henry_conversation import (
    CancelReply,
    ConversationActivated,
    GenerateReply,
    ReplyChunk,
    ReplyGenerationCompleted,
    ReplyGenerationStarted,
    ReplyPhrase,
    UserTurn,
    run_conversation_worker,
)
from henry_conversation.config import ConversationProfile, ConversationPrompts
from henry_conversation.graph import (
    ConversationContext,
    ConversationGraph,
    ConversationNodes,
)
from henry_conversation.reply_segmentation import ReplySegmenter
from henry_conversation.worker import Worker


def context() -> ConversationContext:
    profile = ConversationProfile(
        model="test:model",
        recent_messages=4,
        prompts=ConversationPrompts(
            system="System Polish {conversation_summary}",
            opening="Opening Polish {conversation_summary} {recent_conversation}",
            summary="Summary {conversation_summary} {recent_conversation}",
        ),
    )
    return ConversationContext.from_profile(profile)


def test_config_context_and_events() -> None:
    value = context()
    assert value.model == "test:model"
    assert value.recent_messages == 4

    activation = ConversationActivated()
    turn = UserTurn("Hello")
    assert GenerateReply(activation).input == activation
    assert GenerateReply(turn).input == turn
    assert CancelReply() == CancelReply()
    assert ReplyChunk("a").text == "a"
    assert ReplyPhrase("line").text == "line"
    assert ReplyGenerationStarted() == ReplyGenerationStarted()
    assert ReplyGenerationCompleted() == ReplyGenerationCompleted()

    with pytest.raises(ValidationError):
        ConversationProfile(
            model="",
            recent_messages=1,
            prompts=ConversationPrompts(system="s", opening="o", summary="m"),
        )


def test_reply_segmenter_emits_natural_phrases() -> None:
    segmenter = ReplySegmenter(soft_limit=20, hard_limit=40)

    assert segmenter.feed("Tak. Kolejna wartość to 3.14, a np.") == (
        "Tak.",
        "Kolejna wartość to 3.14,",
    )
    assert segmenter.feed(" ten skrót nie kończy zdania. ") == (
        "a np. ten skrót nie kończy zdania.",
    )
    assert segmenter.feed("To bardzo długa fraza, którą można już wypowiedzieć") == (
        "To bardzo długa fraza,",
    )
    assert segmenter.flush() == ("którą można już wypowiedzieć",)


def test_reply_segmenter_handles_newlines_quotes_limits_and_validation() -> None:
    segmenter = ReplySegmenter(soft_limit=10, hard_limit=12)

    assert segmenter.feed('"Gotowe!"\nNastępna długa fraza bez końca') == (
        '"Gotowe!"',
        "Następna długa",
    )
    assert segmenter.flush() == ("fraza bez końca",)

    protected = ReplySegmenter()
    assert protected.feed("Model U.S. działa przy wersji 3.14 i nazwie x.y") == ()
    assert protected.flush() == ("Model U.S. działa przy wersji 3.14 i nazwie x.y",)

    with pytest.raises(ValueError, match="limits"):
        ReplySegmenter(soft_limit=10, hard_limit=5)
    with pytest.raises(ValueError, match="limits"):
        ReplySegmenter(soft_limit=0)


def test_nodes_build_prompts_and_cache_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        fake = FakeListChatModel(responses=["Opening", "Reply", "New summary"])
        created: list[tuple[str, dict[str, Any]]] = []

        def create_model(model: str, **kwargs: Any) -> FakeListChatModel:
            created.append((model, kwargs))
            return fake

        monkeypatch.setattr(nodes_module, "init_chat_model", create_model)
        nodes = ConversationNodes()
        runtime = type("Runtime", (), {"context": context()})()
        state = {
            "messages": [HumanMessage("Earlier")],
            "summary": "Old subject",
            "input_kind": "activation",
        }

        opening = await nodes.opening(state, runtime)
        reply = await nodes.reply(state, runtime)
        summary = await nodes.summarize(state, runtime)

        assert opening["messages"][0].text == "Opening"
        assert reply["messages"][0].text == "Reply"
        assert summary == {"summary": "New summary", "delivery_context": ""}
        assert created == [
            (
                "test:model",
                {"temperature": 0, "base_url": "http://localhost:11434"},
            )
        ]
        assert nodes._format_messages([]) == "No recent conversation."
        assert "human: Earlier" in nodes._format_messages(state["messages"])

    asyncio.run(scenario())


def test_graph_routes_and_preserves_default_thread_history() -> None:
    async def scenario() -> None:
        nodes = ConversationNodes()
        nodes._models["test:model"] = FakeListChatModel(
            responses=["Welcome", "Answer", "Remembered", "Welcome back"]
        )
        graph = ConversationGraph(nodes, InMemorySaver()).compiled
        config = {"configurable": {"thread_id": "default"}}

        opened = await graph.ainvoke(
            {"messages": [], "input_kind": "activation"},
            config=config,
            context=context(),
        )
        assert [message.text for message in opened["messages"]] == ["Welcome"]

        answered = await graph.ainvoke(
            {
                "messages": [HumanMessage("Question")],
                "input_kind": "user_turn",
            },
            config=config,
            context=context(),
        )
        assert [message.text for message in answered["messages"]] == [
            "Welcome",
            "Question",
            "Answer",
        ]
        assert answered["summary"] == "Remembered"

        reopened = await graph.ainvoke(
            {"messages": [], "input_kind": "activation"},
            config=config,
            context=context(),
        )
        assert reopened["messages"][-1].text == "Welcome back"

        other = await graph.ainvoke(
            {"messages": [], "input_kind": "activation"},
            config={"configurable": {"thread_id": "other"}},
            context=context(),
        )
        assert len(other["messages"]) == 1

    asyncio.run(scenario())


def test_graph_message_stream_identifies_reply_and_summary_nodes() -> None:
    async def scenario() -> None:
        nodes = ConversationNodes()
        nodes._models["test:model"] = FakeListChatModel(
            responses=["Answer\n", "Remembered"]
        )
        graph = ConversationGraph(nodes, InMemorySaver()).compiled
        streamed_nodes: set[str] = set()
        visible = ""

        async for message, metadata in graph.astream(
            {
                "messages": [HumanMessage("Question")],
                "input_kind": "user_turn",
            },
            config={"configurable": {"thread_id": "default"}},
            context=context(),
            stream_mode="messages",
        ):
            node = metadata["langgraph_node"]
            streamed_nodes.add(node)
            if node in ConversationNodes.RESPONSE_NODES:
                visible += message.text

        assert visible == "Answer\n"
        assert streamed_nodes == {ConversationNodes.REPLY, ConversationNodes.SUMMARIZE}

    asyncio.run(scenario())


class FakeCompiled:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    async def astream(self, **kwargs: Any) -> AsyncIterator[tuple[Any, dict]]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        node = (
            ConversationNodes.OPENING
            if kwargs["input"]["input_kind"] == "activation"
            else ConversationNodes.REPLY
        )
        yield HumanMessage("ignored"), {"langgraph_node": node}
        yield AIMessageChunk(content="Hello\n"), {"langgraph_node": node}
        yield AIMessageChunk(content="world"), {"langgraph_node": node}
        yield (
            AIMessageChunk(content="hidden"),
            {"langgraph_node": ConversationNodes.SUMMARIZE},
        )
        yield AIMessageChunk(content=""), {"langgraph_node": node}


class FakeGraph:
    def __init__(self, compiled: FakeCompiled) -> None:
        self.compiled = compiled


def test_worker_streams_activation_turn_chunks_and_lines() -> None:
    async def scenario() -> None:
        bus = EventBus()
        compiled = FakeCompiled()
        worker = Worker(bus, FakeGraph(compiled), context())
        with bus.subscribe(
            ReplyGenerationStarted,
            ReplyChunk,
            ReplyPhrase,
            ReplyGenerationCompleted,
        ) as replies:
            task = asyncio.create_task(worker.run())
            await asyncio.wait_for(_wait_until(lambda: len(bus._subscriptions) == 2), 1)

            bus.publish(GenerateReply(ConversationActivated()))
            first = [await asyncio.wait_for(replies.__anext__(), 1) for _ in range(5)]
            for _ in first:
                replies.task_done()
            assert first == [
                ReplyGenerationStarted(),
                ReplyChunk("Hello\n"),
                ReplyPhrase("Hello"),
                ReplyChunk("world"),
                ReplyPhrase("world"),
            ]
            completed = await asyncio.wait_for(replies.__anext__(), 1)
            replies.task_done()
            assert completed == ReplyGenerationCompleted()

            bus.publish(GenerateReply(UserTurn("Question")))
            while not isinstance(
                await asyncio.wait_for(replies.__anext__(), 1),
                ReplyGenerationCompleted,
            ):
                replies.task_done()
            replies.task_done()

            bus.publish(ShutdownEvent())
            await asyncio.wait_for(task, 1)

        assert compiled.calls[0]["input"] == {
            "delivery_context": "",
            "input_kind": "activation",
            "messages": [],
        }
        assert compiled.calls[1]["input"]["input_kind"] == "user_turn"
        assert compiled.calls[1]["input"]["messages"] == [HumanMessage("Question")]
        assert compiled.calls[0]["config"]["configurable"]["thread_id"] == "default"

    asyncio.run(scenario())


def test_worker_completes_failed_graph_request() -> None:
    async def scenario() -> None:
        bus = EventBus()
        worker = Worker(
            bus, FakeGraph(FakeCompiled(error=RuntimeError("graph"))), context()
        )
        with bus.subscribe(ReplyGenerationStarted, ReplyGenerationCompleted) as replies:
            task = asyncio.create_task(worker.run())
            await asyncio.wait_for(_wait_until(lambda: len(bus._subscriptions) == 2), 1)
            bus.publish(GenerateReply(UserTurn("Question")))
            assert (
                await asyncio.wait_for(replies.__anext__(), 1)
                == ReplyGenerationStarted()
            )
            replies.task_done()
            assert (
                await asyncio.wait_for(replies.__anext__(), 1)
                == ReplyGenerationCompleted()
            )
            replies.task_done()
            with pytest.raises(ExceptionGroup) as raised:
                await asyncio.wait_for(task, 1)
            assert any(
                isinstance(error, RuntimeError) and str(error) == "graph"
                for error in raised.value.exceptions
            )

    asyncio.run(scenario())


def test_worker_cancels_active_reply() -> None:
    class BlockingCompiled(FakeCompiled):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def astream(self, **kwargs: Any) -> AsyncIterator[tuple[Any, dict]]:
            self.calls.append(kwargs)
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
            if False:
                yield AIMessageChunk(""), {}

    async def scenario() -> None:
        bus = EventBus()
        compiled = BlockingCompiled()
        worker = Worker(bus, FakeGraph(compiled), context())

        with bus.subscribe(ReplyGenerationStarted, ReplyGenerationCompleted) as replies:
            task = asyncio.create_task(worker.run())
            await asyncio.wait_for(_wait_until(lambda: len(bus._subscriptions) == 2), 1)
            bus.publish(GenerateReply(UserTurn("Question")))
            assert (
                await asyncio.wait_for(replies.__anext__(), 1)
                == ReplyGenerationStarted()
            )
            replies.task_done()
            await asyncio.wait_for(compiled.started.wait(), 1)

            bus.publish(CancelReply("Delivered phrase."))
            await asyncio.wait_for(compiled.cancelled.wait(), 1)
            assert (
                await asyncio.wait_for(replies.__anext__(), 1)
                == ReplyGenerationCompleted()
            )
            replies.task_done()
            assert worker._delivery_context.startswith(
                "The previous answer was interrupted"
            )
            assert "Delivered phrase." in worker._delivery_context

            bus.publish(ShutdownEvent())
            await asyncio.wait_for(task, 1)

    asyncio.run(scenario())


def test_worker_cancels_queued_reply() -> None:
    async def scenario() -> None:
        worker = Worker(EventBus(), FakeGraph(FakeCompiled()), context())
        worker._graph_queue.put_nowait(UserTurn("Obsolete"))

        await worker._cancel_reply("")

        assert worker._graph_queue.empty()
        await asyncio.wait_for(worker._graph_queue.join(), 1)
        assert "before any part was delivered" in worker._delivery_context
        await worker._stream(UserTurn("Next"))
        assert (
            "before any part was delivered"
            in worker._graph.compiled.calls[0]["input"]["delivery_context"]
        )
        assert worker._delivery_context == ""

    asyncio.run(scenario())


async def _wait_until(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0)


def test_public_runner_builds_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        calls: list[object] = []

        async def fake_run(self: Worker) -> None:
            calls.extend((self._event_bus, self._context, self._graph))

        monkeypatch.setattr(Worker, "run", fake_run)
        bus = EventBus()
        await run_conversation_worker(bus, context())
        assert calls[0] is bus
        assert calls[1] == context()
        assert isinstance(calls[2], ConversationGraph)

    asyncio.run(scenario())


def test_studio_exports_compiled_graph() -> None:
    from henry_conversation.studio import conversation_graph

    assert set(conversation_graph.nodes) == {
        "__start__",
        "opening",
        "reply",
        "summarize",
    }
