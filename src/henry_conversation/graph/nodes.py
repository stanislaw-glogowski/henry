from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from .context import ConversationContext, ConversationRuntime
from .state import ConversationState


class ConversationNodes:
    OPENING = "opening"
    REPLY = "reply"
    SUMMARIZE = "summarize"
    RESPONSE_NODES = frozenset((OPENING, REPLY))

    def __init__(self) -> None:
        self._models: dict[str, BaseChatModel] = {}

    async def opening(
        self,
        state: ConversationState,
        runtime: ConversationRuntime,
    ) -> dict[str, list[AnyMessage] | str]:
        context = runtime.context
        summary = state.get("summary", "")
        messages = state.get("messages", [])[-context.recent_messages :]
        system_prompt = self._format_system_prompt(context, summary)
        opening_prompt = PromptTemplate.from_template(context.opening_prompt).format(
            conversation_summary=summary or "No previous conversation.",
            recent_conversation=self._format_messages(messages),
        )
        response = await self._model(context).ainvoke(
            [
                SystemMessage(content=system_prompt),
                SystemMessage(content=opening_prompt),
                *self._delivery_messages(state),
                *messages,
            ]
        )
        return {"messages": [response], "delivery_context": ""}

    async def reply(
        self,
        state: ConversationState,
        runtime: ConversationRuntime,
    ) -> dict[str, list[AnyMessage] | str]:
        context = runtime.context
        summary = state.get("summary", "")
        messages = state["messages"][-context.recent_messages :]
        response = await self._model(context).ainvoke(
            [
                SystemMessage(content=self._format_system_prompt(context, summary)),
                *self._delivery_messages(state),
                *messages,
            ]
        )
        return {"messages": [response]}

    async def summarize(
        self,
        state: ConversationState,
        runtime: ConversationRuntime,
    ) -> dict[str, str]:
        context = runtime.context
        summary_prompt = PromptTemplate.from_template(context.summary_prompt).format(
            conversation_summary=state.get("summary", "") or "No previous summary.",
            recent_conversation=self._format_messages(
                state["messages"][-context.recent_messages :]
            ),
        )
        response = await self._model(context).ainvoke(
            [*self._delivery_messages(state), SystemMessage(content=summary_prompt)]
        )
        return {"summary": response.text, "delivery_context": ""}

    def _model(self, context: ConversationContext) -> BaseChatModel:
        if model := self._models.get(context.model):
            return model

        model = init_chat_model(
            context.model,
            temperature=0,
            base_url="http://localhost:11434",
        )
        self._models[context.model] = model
        return model

    @staticmethod
    def _format_system_prompt(
        context: ConversationContext,
        summary: str,
    ) -> str:
        return PromptTemplate.from_template(context.system_prompt).format(
            conversation_summary=summary or "No previous conversation.",
        )

    @staticmethod
    def _format_messages(messages: list[AnyMessage]) -> str:
        if not messages:
            return "No recent conversation."

        return "\n".join(
            f"{message.type}: {message.text}" for message in messages if message.text
        )

    @staticmethod
    def _delivery_messages(state: ConversationState) -> list[SystemMessage]:
        context = state.get("delivery_context", "")
        if not context:
            return []
        return [SystemMessage(content=context)]
