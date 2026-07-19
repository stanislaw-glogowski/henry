import asyncio
from dataclasses import dataclass

from loguru import logger

from ..domain import AssistantReply, Conversation, SpeechTranscription
from ..events import PipelineStage, PipelineStageChanged, PipelineStageStatus
from ..ports import AppEventSink, LLModel, TelemetrySink


@dataclass(frozen=True, slots=True)
class ConversationConfig:
    system_prompt: str


class ConversationService:
    def __init__(self, model: LLModel, config: ConversationConfig) -> None:
        self._model = model
        self._config = config
        self._conversation = Conversation(system_prompt=config.system_prompt)
        self._logger = logger.bind(component="ConversationService")

    async def run(
        self,
        transcriptions: asyncio.Queue[SpeechTranscription],
        replies: asyncio.Queue[AssistantReply],
        events: AppEventSink,
        telemetry: TelemetrySink,
    ) -> None:
        self._logger.debug("Running")

        events.publish(
            PipelineStageChanged(PipelineStage.CONVERSATION, PipelineStageStatus.READY)
        )

        while True:
            transcription = await transcriptions.get()

            try:
                events.publish(
                    PipelineStageChanged(
                        PipelineStage.CONVERSATION, PipelineStageStatus.STARTED
                    )
                )

                self._conversation.add_user_message(transcription.text)

                self._logger.trace("Sending REQUEST")

                text = await self._model.generate_reply(
                    self._conversation.messages,
                )

                self._logger.trace("Request COMPLETED, reply: '{}'", text)

                events.publish(
                    PipelineStageChanged(
                        PipelineStage.CONVERSATION, PipelineStageStatus.COMPLETED
                    )
                )

                self._conversation.add_assistant_message(text)

                await replies.put(
                    AssistantReply(
                        text=text,
                    ),
                )

            finally:
                transcriptions.task_done()
