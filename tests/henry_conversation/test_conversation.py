import asyncio
import sys
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

import henry_conversation.model as model_module
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
from henry_conversation.config import ConversationSettings
from henry_conversation.graph import (
    ConversationContext,
    ConversationGraph,
    ConversationNodes,
    ResponseMode,
    ResponseRouter,
    TurnIntent,
)
from henry_conversation.model import (
    ConversationMessage,
    ConversationRole,
    LanguageModelChunk,
    LanguageModelRequest,
    LanguageModelRole,
    LanguageModelService,
    get_language_model,
)
from henry_conversation.model.adapters.langchain import LangChainLanguageModel
from henry_conversation.model.adapters.mlx import MLXLanguageModel
from henry_conversation.model.config import (
    LangChainModelProfile,
    LangChainSettings,
    MLXModelProfile,
    MLXSettings,
)
from henry_conversation.profile import (
    ConversationProfile,
    ConversationPrompts,
    ConversationReactions,
    ProfilePreparation,
)
from henry_conversation.reply import ConversationTextChunk, ReplySegmenter
from henry_conversation.worker import Worker
from tests.support import FakeLanguageModel


def profile() -> ConversationProfile:
    return ConversationProfile(
        models={
            "fast": {"model_id": "test/fast", "max_tokens": 96},
            "detailed": {"model_id": "test/detailed", "max_tokens": 256},
            "classifier": {"model_id": "test/classifier", "max_tokens": 4},
        },
        recent_messages=4,
        prompts=ConversationPrompts(
            system="System {conversation_summary}",
            opening="Opening {conversation_summary} {recent_conversation}",
            summary="Summary {conversation_summary} {recent_conversation}",
        ),
    )


def context(delay: float = 0.5) -> ConversationContext:
    return ConversationContext.from_profile(
        profile(), ConversationSettings(acknowledgement_delay=delay)
    )


def test_configuration_domain_events_and_routing() -> None:
    value = profile()
    assert value.models_langchain.fast.model_id == "test/fast"
    mlx_value = value.model_copy(
        update={
            "models": {
                **value.models,
                "fast": {**value.models["fast"], "top_k": 12},
            }
        }
    )
    assert mlx_value.models_mlx.fast.top_k == 12
    with pytest.raises(ValidationError, match="top_k"):
        _ = mlx_value.models_langchain
    assert isinstance(ConversationSettings().language_model, MLXSettings)
    assert isinstance(
        ConversationSettings.model_validate(
            {"language_model": {"adapter": "mlx"}}
        ).language_model,
        MLXSettings,
    )
    with pytest.raises(ValidationError, match="model_id"):
        _ = ConversationProfile(
            models={"fast": {}, "detailed": {}},
            prompts=value.prompts,
        ).models_langchain
    legacy = value.model_copy(
        update={
            "models": {
                "fast": {"langchain": "test:fast", "mlx": "test/fast"},
                "detailed": {
                    "langchain": "test:detailed",
                    "mlx": "test/detailed",
                },
            }
        }
    )
    with pytest.raises(ValidationError, match="model_id"):
        _ = legacy.models_langchain
    with pytest.raises(ValidationError):
        ConversationReactions(wake=("",))
    with pytest.raises(ValidationError):
        ConversationProfile(
            models=value.models,
            recent_messages=1,
            prompts=value.prompts,
        )

    activation = ConversationActivated()
    turn = UserTurn("Hello")
    assert GenerateReply(activation).input == activation
    assert GenerateReply(turn).input == turn
    assert CancelReply() == CancelReply()
    assert ReplyChunk(1, "a").text == "a"
    assert ReplyPhrase(1, 1, "line").text == "line"
    assert ReplyGenerationStarted(1) == ReplyGenerationStarted(1)
    assert ReplyGenerationCompleted(1) == ReplyGenerationCompleted(1)

    router = ResponseRouter()
    assert router.plan("").intent is TurnIntent.NO_RESPONSE
    assert router.plan("short question").mode is ResponseMode.FAST
    detailed = router.plan(" ".join(f"word{index}" for index in range(24)))
    assert detailed.mode is ResponseMode.DETAILED
    assert detailed.acknowledge
    assert router.plan("one two three four five six seven eight? nine ten? ").mode is (
        ResponseMode.DETAILED
    )
    assert router.plan(" ".join("word" for _ in range(18))).mode is ResponseMode.FAST
    assert router.is_ambiguous(" ".join("word" for _ in range(18)))
    assert router.is_ambiguous("tell me a longer story")
    assert not router.is_ambiguous("what time is it?")
    assert router.classified_plan(" detailed ").mode is ResponseMode.DETAILED
    assert router.classified_plan("unexpected").mode is ResponseMode.FAST


def test_reply_segmenter_emits_natural_phrases_and_validates_limits() -> None:
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

    quoted = ReplySegmenter(soft_limit=10, hard_limit=12)
    assert quoted.feed('"Gotowe!"\nNastępna długa fraza bez końca') == (
        '"Gotowe!"',
        "Następna długa",
    )
    assert quoted.flush() == ("fraza bez końca",)
    protected = ReplySegmenter()
    assert protected.feed("Model U.S. działa przy wersji 3.14 i nazwie x.y") == ()
    assert protected.flush() == ("Model U.S. działa przy wersji 3.14 i nazwie x.y",)
    with pytest.raises(ValueError, match="limits"):
        ReplySegmenter(soft_limit=10, hard_limit=5)
    with pytest.raises(ValueError, match="limits"):
        ReplySegmenter(soft_limit=0)


def test_language_model_service_owns_resources_and_streams() -> None:
    async def scenario() -> None:
        adapter = FakeLanguageModel("Answer")
        service = LanguageModelService(adapter)
        request = LanguageModelRequest(
            LanguageModelRole.FAST,
            (ConversationMessage(ConversationRole.USER, "Question"),),
        )
        with pytest.raises(RuntimeError, match="not started"):
            await service.prepare(LanguageModelRole.FAST)
        async with service:
            await service.prepare(LanguageModelRole.FAST)
            chunks = [chunk async for chunk in service.generate(request)]
        assert "".join(chunk.content for chunk in chunks) == "Answer"
        assert adapter.requests == [request]
        thread_ids = {thread_id for _, thread_id in adapter.operations}
        assert len(thread_ids) == 1

        await service.stop()

    asyncio.run(scenario())


def test_language_model_service_propagates_generation_error() -> None:
    class FailingModel(FakeLanguageModel):
        def generate(self, request: LanguageModelRequest):
            raise RuntimeError("generation failed")
            yield

    async def scenario() -> None:
        async with LanguageModelService(FailingModel()) as service:
            with pytest.raises(RuntimeError, match="generation failed"):
                _ = [
                    chunk
                    async for chunk in service.generate(
                        LanguageModelRequest(LanguageModelRole.FAST, ())
                    )
                ]

    asyncio.run(scenario())


def test_profile_preparation_generates_rotating_reactions() -> None:
    async def scenario() -> None:
        model = FakeLanguageModel()
        async with LanguageModelService(model) as service:
            preparation = ProfilePreparation(
                service,
                ConversationReactions(
                    wake=("Wake one.", "Wake two."),
                    wait=("Wait one.", "Wait two."),
                ),
            )
            await preparation.prepare()
            assert [preparation.next_reaction() for _ in range(3)] == [
                "Wait one.",
                "Wait two.",
                "Wait one.",
            ]
            assert [preparation.next_wake_reaction() for _ in range(3)] == [
                "Wake one.",
                "Wake two.",
                "Wake one.",
            ]
        assert any(
            operation == f"prepare:{LanguageModelRole.FAST}"
            for operation, _ in model.operations
        )

    asyncio.run(scenario())


def test_graph_routes_streams_and_preserves_thread_history() -> None:
    async def scenario() -> None:
        adapter = FakeLanguageModel("Welcome", "Answer", "Remembered", "Welcome back")
        async with LanguageModelService(adapter) as service:
            graph = ConversationGraph(
                ConversationNodes(service), InMemorySaver()
            ).compiled
            config = {"configurable": {"thread_id": "default"}}
            opened = await graph.ainvoke(
                {"messages": [], "input_kind": "activation"},
                config=config,
                context=context(),
            )
            assert [message.text for message in opened["messages"]] == ["Welcome"]

            visible: list[ConversationTextChunk] = []
            async for event in graph.astream(
                {
                    "messages": [HumanMessage("Question")],
                    "input_kind": "user_turn",
                },
                config=config,
                context=context(),
                stream_mode="custom",
            ):
                visible.append(event)
            assert "".join(event.content for event in visible) == "Answer"

            state = await graph.aget_state(config)
            assert state.values["summary"] == "Remembered"
            reopened = await graph.ainvoke(
                {"messages": [], "input_kind": "activation"},
                config=config,
                context=context(),
            )
            assert reopened["messages"][-1].text == "Welcome back"

        assert adapter.requests[0].role is LanguageModelRole.FAST
        assert adapter.requests[1].role is LanguageModelRole.FAST
        assert adapter.requests[2].role is LanguageModelRole.FAST
        assert adapter.requests[0].messages[0].role is ConversationRole.SYSTEM
        assert adapter.requests[3].messages[0].role is ConversationRole.SYSTEM
        assert adapter.requests[0].messages[0] != adapter.requests[3].messages[0]
        assert all(
            message.role is not ConversationRole.SYSTEM
            for message in adapter.requests[0].messages[1:]
        )

    asyncio.run(scenario())


def test_graph_emits_prepared_acknowledgement_for_slow_detailed_reply() -> None:
    class Prepared:
        def next_wake_reaction(self) -> None:
            return None

        def next_reaction(self) -> str:
            return "Reaction."

    async def scenario() -> None:
        long_question = " ".join(f"word{index}" for index in range(24))
        adapter = FakeLanguageModel("Detailed", "Summary")
        async with LanguageModelService(adapter) as service:
            graph = ConversationGraph(
                ConversationNodes(service, profile_preparation=Prepared())
            ).compiled
            events = [
                event
                async for event in graph.astream(
                    {
                        "messages": [HumanMessage(long_question)],
                        "input_kind": "user_turn",
                    },
                    context=context(delay=0),
                    stream_mode="custom",
                )
            ]
        assert events[0] == ConversationTextChunk("Reaction.\n", True)
        assert "".join(event.content for event in events[1:]) == "Detailed"
        assert adapter.requests[0].role is LanguageModelRole.DETAILED

    asyncio.run(scenario())


def test_graph_uses_optional_classifier_for_ambiguous_turn() -> None:
    async def scenario() -> None:
        question = " ".join(f"word{index}" for index in range(18))
        adapter = FakeLanguageModel("DETAILED", "Answer", "Summary")
        settings = ConversationSettings(
            classify_ambiguous=True,
            acknowledgement_delay=10,
        )
        async with LanguageModelService(adapter) as service:
            graph = ConversationGraph(ConversationNodes(service)).compiled
            events = [
                event
                async for event in graph.astream(
                    {
                        "messages": [HumanMessage(question)],
                        "input_kind": "user_turn",
                    },
                    context=ConversationContext.from_profile(profile(), settings),
                    stream_mode="custom",
                )
            ]
        assert "".join(event.content for event in events) == "Answer"
        assert [request.role for request in adapter.requests] == [
            LanguageModelRole.CLASSIFIER,
            LanguageModelRole.DETAILED,
            LanguageModelRole.FAST,
        ]

    asyncio.run(scenario())


def test_graph_uses_prepared_wake_reaction_without_model_generation() -> None:
    class Prepared:
        def next_wake_reaction(self) -> str:
            return "Listening."

        def next_reaction(self) -> None:
            return None

    async def scenario() -> None:
        adapter = FakeLanguageModel()
        async with LanguageModelService(adapter) as service:
            graph = ConversationGraph(
                ConversationNodes(service, profile_preparation=Prepared())
            ).compiled
            events = [
                event
                async for event in graph.astream(
                    {"messages": [], "input_kind": "activation"},
                    context=context(),
                    stream_mode="custom",
                )
            ]
        assert events == [ConversationTextChunk("Listening.\n", True)]
        assert adapter.requests == []

    asyncio.run(scenario())


class FakeCompiled:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error = error

    async def astream(self, **kwargs: Any) -> AsyncIterator[ConversationTextChunk]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        yield SimpleNamespace(content="ignored")
        yield ConversationTextChunk("")
        yield ConversationTextChunk("Hello\n")
        yield ConversationTextChunk("world")


class FakeGraph:
    def __init__(self, compiled: FakeCompiled) -> None:
        self.compiled = compiled


def test_worker_streams_chunks_phrases_and_inputs() -> None:
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
            received = []
            while not any(
                isinstance(event, ReplyGenerationCompleted) for event in received
            ):
                event = await asyncio.wait_for(replies.__anext__(), 1)
                received.append(event)
                replies.task_done()
            assert received == [
                ReplyGenerationStarted(1),
                ReplyChunk(1, "Hello\n"),
                ReplyPhrase(1, 1, "Hello"),
                ReplyChunk(1, "world"),
                ReplyPhrase(1, 2, "world"),
                ReplyGenerationCompleted(1),
            ]
            bus.publish(GenerateReply(UserTurn("Question")))
            while not isinstance(
                event := await asyncio.wait_for(replies.__anext__(), 1),
                ReplyGenerationCompleted,
            ):
                replies.task_done()
            replies.task_done()
            bus.publish(ShutdownEvent())
            await asyncio.wait_for(task, 1)

        assert compiled.calls[0]["input"]["input_kind"] == "activation"
        assert compiled.calls[1]["input"]["messages"] == [HumanMessage("Question")]
        assert compiled.calls[0]["stream_mode"] == "custom"

    asyncio.run(scenario())


def test_worker_cancels_active_and_queued_replies() -> None:
    class BlockingCompiled(FakeCompiled):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def astream(self, **kwargs: Any) -> AsyncIterator[ConversationTextChunk]:
            self.calls.append(kwargs)
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
            if False:
                yield ConversationTextChunk("")

    async def scenario() -> None:
        bus = EventBus()
        compiled = BlockingCompiled()
        worker = Worker(bus, FakeGraph(compiled), context())
        with bus.subscribe(ReplyGenerationStarted, ReplyGenerationCompleted) as replies:
            task = asyncio.create_task(worker.run())
            await asyncio.wait_for(_wait_until(lambda: len(bus._subscriptions) == 2), 1)
            bus.publish(GenerateReply(UserTurn("Question")))
            assert isinstance(await replies.__anext__(), ReplyGenerationStarted)
            replies.task_done()
            await compiled.started.wait()
            bus.publish(CancelReply("Delivered phrase."))
            await asyncio.wait_for(compiled.cancelled.wait(), 1)
            assert isinstance(await replies.__anext__(), ReplyGenerationCompleted)
            replies.task_done()
            assert "Delivered phrase." in worker._delivery_context
            bus.publish(ShutdownEvent())
            await asyncio.wait_for(task, 1)

        queued = Worker(EventBus(), FakeGraph(FakeCompiled()), context())
        queued._graph_queue.put_nowait(UserTurn("Obsolete"))
        await queued._cancel_reply("")
        await queued._graph_queue.join()
        await queued._stream(1, UserTurn("Next"))
        assert (
            "before any part was delivered"
            in (queued._graph.compiled.calls[0]["input"]["delivery_context"])
        )

    asyncio.run(scenario())


def test_worker_propagates_graph_failure_after_completion() -> None:
    async def scenario() -> None:
        bus = EventBus()
        worker = Worker(
            bus, FakeGraph(FakeCompiled(error=RuntimeError("graph"))), context()
        )
        with bus.subscribe(ReplyGenerationStarted, ReplyGenerationCompleted) as replies:
            task = asyncio.create_task(worker.run())
            await asyncio.wait_for(_wait_until(lambda: len(bus._subscriptions) == 2), 1)
            bus.publish(GenerateReply(UserTurn("Question")))
            assert isinstance(await replies.__anext__(), ReplyGenerationStarted)
            replies.task_done()
            assert isinstance(await replies.__anext__(), ReplyGenerationCompleted)
            replies.task_done()
            with pytest.raises(ExceptionGroup) as raised:
                await task
            assert any(
                isinstance(error, RuntimeError) and str(error) == "graph"
                for error in raised.value.exceptions
            )

    asyncio.run(scenario())


def test_worker_treats_profile_preparation_as_optional() -> None:
    class Preparation:
        def __init__(self, error: Exception | None = None) -> None:
            self.error = error
            self.called = False

        async def prepare(self) -> None:
            self.called = True
            if self.error is not None:
                raise self.error

    async def scenario() -> None:
        worker = Worker(EventBus(), FakeGraph(FakeCompiled()), context())
        await worker._prepare_profile()

        successful = Preparation()
        worker._profile_preparation = successful
        await worker._prepare_profile()
        assert successful.called

        failing = Preparation(RuntimeError("profile preparation failed"))
        worker._profile_preparation = failing
        with pytest.raises(RuntimeError, match="profile preparation failed"):
            await worker._prepare_profile()
        assert failing.called

    asyncio.run(scenario())


def test_langchain_adapter_maps_messages_and_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeChat:
        def stream(self, messages):
            assert [message.type for message in messages] == ["system", "human"]
            yield SimpleNamespace(text="answer")

    def init_chat_model(model: str, **kwargs: Any) -> FakeChat:
        calls.append((model, kwargs))
        return FakeChat()

    monkeypatch.setattr("langchain.chat_models.init_chat_model", init_chat_model)
    adapter = LangChainLanguageModel(
        profile().models_langchain,
        LangChainSettings(),
    )
    with adapter:
        adapter.prepare(LanguageModelRole.FAST)
        chunks = list(
            adapter.generate(
                LanguageModelRequest(
                    LanguageModelRole.FAST,
                    (
                        ConversationMessage(ConversationRole.SYSTEM, "system"),
                        ConversationMessage(ConversationRole.USER, "question"),
                    ),
                )
            )
        )
    assert chunks == [LanguageModelChunk("answer")]
    assert calls[0][0] == "test/fast"
    assert calls[0][1]["num_predict"] == 96
    assert calls[0][1]["reasoning"] is False

    ollama_profile = profile().models_langchain.model_copy(
        update={
            "fast": LangChainModelProfile(
                model_id="ollama:gpt-oss:20b",
                max_tokens=96,
            )
        }
    )
    with LangChainLanguageModel(
        ollama_profile,
        LangChainSettings(base_url="http://test.local"),
    ) as ollama_adapter:
        ollama_adapter.prepare(LanguageModelRole.FAST)
    assert calls[1][1]["reasoning"] == "low"
    assert calls[1][1]["base_url"] == "http://test.local"


def test_mlx_adapter_loads_lazily_and_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {
        "templates": [],
        "cache_count": 0,
        "models": [],
        "generations": [],
    }

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls["templates"].append((messages, kwargs))
            return "system-prompt" if len(messages) > 1 else "system"

        def encode(self, text):
            return [ord(character) for character in text]

    mlx_lm = ModuleType("mlx_lm")
    loaded_model = "loaded-model"

    def load(model):
        calls["models"].append(model)
        return loaded_model, Tokenizer()

    mlx_lm.load = load

    def stream_generate(*args, **kwargs):
        calls["generations"].append(kwargs)
        yield SimpleNamespace(text="chunk")

    mlx_lm.stream_generate = stream_generate
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx_core.array = lambda value: value
    mlx.core = mlx_core
    generate = ModuleType("mlx_lm.generate")
    generate.generate_step = lambda *args, **kwargs: iter(())
    models = ModuleType("mlx_lm.models")
    cache = ModuleType("mlx_lm.models.cache")

    def make_prompt_cache(model):
        calls["cache_count"] += 1
        return {"model": model}

    cache.make_prompt_cache = make_prompt_cache
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **kwargs: calls.setdefault("sampler", kwargs)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    models = profile().models_mlx.model_copy(
        update={
            "fast": MLXModelProfile(model_id="test/detailed", max_tokens=96),
        }
    )
    adapter = MLXLanguageModel(models)
    with adapter:
        chunks = list(
            adapter.generate(
                LanguageModelRequest(
                    LanguageModelRole.DETAILED,
                    (
                        ConversationMessage(ConversationRole.SYSTEM, "system"),
                        ConversationMessage(ConversationRole.USER, "question"),
                    ),
                )
            )
        )
        _ = list(
            adapter.generate(
                LanguageModelRequest(
                    LanguageModelRole.DETAILED,
                    (
                        ConversationMessage(ConversationRole.SYSTEM, "system"),
                        ConversationMessage(ConversationRole.USER, "question"),
                    ),
                )
            )
        )
        _ = list(
            adapter.generate(
                LanguageModelRequest(
                    LanguageModelRole.FAST,
                    (ConversationMessage(ConversationRole.USER, "question"),),
                )
            )
        )
    assert chunks == [LanguageModelChunk("chunk")]
    assert calls["models"] == ["test/detailed"]
    assert calls["generations"][0]["max_tokens"] == 256
    assert calls["templates"][0][0] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
    assert calls["templates"][0][1]["enable_thinking"] is False
    assert calls["cache_count"] == 1
    assert calls["generations"][0]["prompt"] == [
        ord(character) for character in "-prompt"
    ]
    assert calls["generations"][0]["prompt_cache"] == {"model": loaded_model}


def test_mlx_adapter_generates_without_unsupported_system_prefix_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated: dict[str, Any] = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if len(messages) == 1:
                raise RuntimeError("A user message is required")
            return "complete-prompt"

        def encode(self, text):
            return [1]

    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.load = lambda model: (object(), Tokenizer())

    def stream_generate(*args, **kwargs):
        generated.update(kwargs)
        yield SimpleNamespace(text="answer")

    mlx_lm.stream_generate = stream_generate
    mlx = ModuleType("mlx")
    mlx_core = ModuleType("mlx.core")
    mlx_core.array = lambda value: value
    mlx.core = mlx_core
    generate = ModuleType("mlx_lm.generate")
    generate.generate_step = lambda *args, **kwargs: iter(())
    models = ModuleType("mlx_lm.models")
    cache = ModuleType("mlx_lm.models.cache")
    cache.make_prompt_cache = lambda model: object()
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    adapter = MLXLanguageModel(profile().models_mlx)
    with adapter:
        chunks = list(
            adapter.generate(
                LanguageModelRequest(
                    LanguageModelRole.FAST,
                    (
                        ConversationMessage(ConversationRole.SYSTEM, "system"),
                        ConversationMessage(ConversationRole.USER, "question"),
                    ),
                )
            )
        )

    assert chunks == [LanguageModelChunk("answer")]
    assert generated["prompt"] == "complete-prompt"
    assert generated["prompt_cache"] is None


def test_adapter_factory_and_public_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(
        get_language_model(profile(), LangChainSettings()),
        LangChainLanguageModel,
    )
    assert isinstance(
        get_language_model(profile(), MLXSettings()),
        MLXLanguageModel,
    )
    with pytest.raises(ValueError, match="Unsupported"):
        get_language_model(profile(), SimpleNamespace(adapter="unknown"))
    incomplete = profile().model_copy(
        update={
            "models": {"fast": {"model_id": "only"}},
        }
    )
    with pytest.raises(ValidationError, match="detailed"):
        get_language_model(incomplete, MLXSettings())
    no_classifier = profile().model_copy(
        update={
            "models": {
                "fast": {"model_id": "test/fast"},
                "detailed": {"model_id": "test/detailed"},
            }
        }
    )
    with pytest.raises(ValueError, match="requires a classifier"):
        get_language_model(
            no_classifier,
            LangChainSettings(),
            require_classifier=True,
        )

    async def scenario() -> None:
        calls: list[Worker] = []
        fake_model = FakeLanguageModel("One.\nTwo.")

        async def fake_run(self: Worker) -> None:
            calls.append(self)

        monkeypatch.setattr(Worker, "run", fake_run)
        monkeypatch.setattr(
            model_module,
            "get_language_model",
            lambda *_, **__: fake_model,
        )
        await run_conversation_worker(EventBus(), profile(), ConversationSettings())
        assert len(calls) == 1
        assert isinstance(calls[0]._graph, ConversationGraph)

    asyncio.run(scenario())


def test_studio_exports_compiled_graph() -> None:
    from henry_conversation.studio import conversation_graph

    async def scenario() -> None:
        async with conversation_graph() as graph:
            assert set(graph.nodes) == {
                "__start__",
                "opening",
                "reply",
                "summarize",
            }

    asyncio.run(scenario())


async def _wait_until(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0)
