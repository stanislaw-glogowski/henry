from loguru import logger

from henry_common.events import EventBus, ShutdownEvent
from henry_conversation.events import (
    ConversationActivated,
    GenerateReply,
    ReplyGenerationCompleted,
    ReplyGenerationStarted,
    ReplyPhrase,
    UserTurn,
)
from henry_speech.events import InteractionTimingObserved, WakeWordObserved


async def run_event_logger(event_bus: EventBus) -> None:
    with event_bus.subscribe(
        GenerateReply,
        ReplyGenerationStarted,
        ReplyPhrase,
        ReplyGenerationCompleted,
        InteractionTimingObserved,
        WakeWordObserved,
        ShutdownEvent,
    ) as events:
        async for event in events:
            try:
                match event:
                    case GenerateReply(ConversationActivated()):
                        logger.info("Wake word activated the conversation")
                    case GenerateReply(UserTurn(text)):
                        logger.info("User: {}", text)
                    case ReplyGenerationStarted():
                        logger.debug("Generating response")
                    case ReplyPhrase(text):
                        logger.info("Henry: {}", text)
                    case ReplyGenerationCompleted():
                        logger.debug("Response completed")
                    case InteractionTimingObserved(stage, elapsed_ms):
                        logger.debug(
                            "Interaction timing: stage='{}', elapsed_ms={:.1f}",
                            stage,
                            elapsed_ms,
                        )
                    case WakeWordObserved(detected=True):
                        logger.debug("Wake word detected")
                    case ShutdownEvent():
                        logger.debug("Shutdown requested")
                        return
            finally:
                events.task_done()
