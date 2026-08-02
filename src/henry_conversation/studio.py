from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from henry_conversation.graph import (
    ConversationContext,
    ConversationGraph,
    ConversationNodes,
)
from henry_conversation.model import LanguageModelService
from henry_conversation.model.adapters.langchain import LangChainLanguageModel
from henry_conversation.model.config import (
    LangChainModelProfile,
    LangChainModelsProfile,
    LangChainSettings,
)

_STUDIO_MODELS = LangChainModelsProfile(
    fast=LangChainModelProfile(model_id="ollama:gpt-oss:20b"),
    detailed=LangChainModelProfile(model_id="ollama:gpt-oss:20b"),
)


@asynccontextmanager
async def conversation_graph() -> AsyncGenerator[
    CompiledStateGraph[Any, ConversationContext, Any, Any]
]:
    """Own the model service for the lifetime of a LangGraph Studio run."""

    adapter = LangChainLanguageModel(_STUDIO_MODELS, LangChainSettings())
    async with LanguageModelService(adapter) as service:
        yield ConversationGraph(nodes=ConversationNodes(service)).compiled
