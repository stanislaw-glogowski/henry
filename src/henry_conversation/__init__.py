from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from henry_common.events import EventBus

from .config import ConversationProfile, ConversationReactions, ConversationSettings
from .domain import (
    ConversationMessage,
    ConversationRole,
    ConversationTextChunk,
    LanguageModelChunk,
    LanguageModelRequest,
    LanguageModelRole,
    ResponseMode,
    ResponsePlan,
    TurnIntent,
)
from .events import (
    CancelReply,
    ConversationActivated,
    GenerateReply,
    ReplyChunk,
    ReplyGenerationCompleted,
    ReplyGenerationStarted,
    ReplyPhrase,
    UserTurn,
)

__all__ = [
    "CancelReply",
    "ConversationActivated",
    "ConversationMessage",
    "ConversationProfile",
    "ConversationReactions",
    "ConversationRole",
    "ConversationSettings",
    "ConversationTextChunk",
    "GenerateReply",
    "LanguageModelChunk",
    "LanguageModelRequest",
    "LanguageModelRole",
    "ReplyChunk",
    "ReplyGenerationCompleted",
    "ReplyGenerationStarted",
    "ReplyPhrase",
    "ResponseMode",
    "ResponsePlan",
    "TurnIntent",
    "UserTurn",
    "run_conversation_worker",
]


async def run_conversation_worker(
    event_bus: EventBus,
    profile: ConversationProfile,
    settings: ConversationSettings,
) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from .graph import ConversationContext, ConversationGraph, ConversationNodes
    from .model import LanguageModelService, get_language_model
    from .preparation import ProfilePreparation
    from .worker import Worker

    context = ConversationContext.from_profile(profile, settings)
    async with LanguageModelService(get_language_model(profile, settings)) as service:
        preparation = ProfilePreparation(service, profile.reactions)
        graph = ConversationGraph(
            nodes=ConversationNodes(service, profile_preparation=preparation),
            checkpointer=InMemorySaver(),
        )
        await Worker(
            event_bus=event_bus,
            graph=graph,
            context=context,
            profile_preparation=preparation,
        ).run()
