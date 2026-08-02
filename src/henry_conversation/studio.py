from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from henry_conversation.graph import (
    ConversationContext,
    ConversationGraph,
    ConversationNodes,
)
from henry_conversation.model import LanguageModelService, get_language_model
from henry_resources import LocalStore

_DEFAULT_PROFILE_ID = "default"


@asynccontextmanager
async def conversation_graph() -> AsyncGenerator[
    CompiledStateGraph[Any, ConversationContext, Any, Any]
]:
    """Own the model service for the lifetime of a LangGraph Studio run."""

    store = LocalStore()
    profile = store.load_profile(_DEFAULT_PROFILE_ID).conversation
    settings = store.load_settings().conversation
    adapter = get_language_model(
        profile,
        settings.language_model,
        require_classifier=settings.classify_ambiguous,
    )
    async with LanguageModelService(adapter) as service:
        yield ConversationGraph(nodes=ConversationNodes(service)).compiled
