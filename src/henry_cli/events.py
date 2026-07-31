from loguru import logger

from henry_common.events import EventBus, ShutdownEvent
from henry_conversation.events import (
    ConversationActivated,
    GenerateReply,
    ReplyCompleted,
    ReplyLine,
    ReplyStarted,
    UserTurn,
)
from henry_speech.events import WakeWordObserved


async def run_event_logger(event_bus: EventBus) -> None:
    with event_bus.subscribe(
        GenerateReply,
        ReplyStarted,
        ReplyLine,
        ReplyCompleted,
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
                    case ReplyStarted():
                        logger.debug("Generating response")
                    case ReplyLine(text):
                        logger.info("Henry: {}", text)
                    case ReplyCompleted():
                        logger.debug("Response completed")
                    case WakeWordObserved(detected=True):
                        logger.debug("Wake word detected")
                    case ShutdownEvent():
                        logger.debug("Shutdown requested")
                        return
            finally:
                events.task_done()
