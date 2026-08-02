from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.graph.state import CompiledStateGraph

from henry_conversation.config import LanguageModelProfile, LanguageModelsProfile
from henry_conversation.graph import ConversationGraph, ConversationNodes
from henry_conversation.model.adapters.langchain import LangChainLanguageModel
from henry_conversation.model.service import LanguageModelService

_STUDIO_MODELS = LanguageModelsProfile(
    fast=LanguageModelProfile(langchain="ollama:gpt-oss:20b"),
    detailed=LanguageModelProfile(langchain="ollama:gpt-oss:20b"),
)


@asynccontextmanager
async def conversation_graph() -> AsyncIterator[CompiledStateGraph]:
    """Own the model service for the lifetime of a LangGraph Studio run."""

    async with LanguageModelService(LangChainLanguageModel(_STUDIO_MODELS)) as service:
        yield ConversationGraph(nodes=ConversationNodes(service)).compiled
