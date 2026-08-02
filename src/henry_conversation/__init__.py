from henry_common.events import EventBus

from .config import ConversationSettings
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
from .graph import ResponseMode, ResponsePlan, TurnIntent
from .model import (
    ConversationMessage,
    ConversationRole,
    LanguageModelChunk,
    LanguageModelRequest,
    LanguageModelRole,
)
from .profile import ConversationProfile, ConversationReactions
from .reply import ConversationTextChunk

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
    from .profile import ProfilePreparation
    from .worker import Worker

    language_model = get_language_model(
        profile,
        settings.model,
        require_classifier=settings.classify_ambiguous,
    )
    context = ConversationContext.from_profile(profile, settings)
    async with LanguageModelService(language_model) as service:
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
