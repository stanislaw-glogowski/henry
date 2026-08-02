import asyncio
from contextlib import suppress

from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.config import get_stream_writer

from ..model import (
    ConversationMessage,
    ConversationRole,
    LanguageModelRequest,
    LanguageModelRole,
    LanguageModelService,
)
from ..profile import ProfilePreparation
from ..reply import ConversationTextChunk
from .context import ConversationContext, ConversationRuntime
from .routing import ResponseMode, ResponsePlan, ResponseRouter
from .state import ConversationState


class ConversationNodes:
    OPENING = "opening"
    REPLY = "reply"
    SUMMARIZE = "summarize"

    def __init__(
        self,
        language_model: LanguageModelService,
        response_router: ResponseRouter | None = None,
        profile_preparation: ProfilePreparation | None = None,
    ) -> None:
        self._language_model = language_model
        self._response_router = response_router or ResponseRouter()
        self._profile_preparation = profile_preparation

    async def opening(
        self,
        state: ConversationState,
        runtime: ConversationRuntime,
    ) -> dict[str, list[AnyMessage] | str]:
        context = runtime.context
        if self._profile_preparation is not None and (
            reaction := self._profile_preparation.next_wake_reaction()
        ):
            get_stream_writer()(ConversationTextChunk(f"{reaction}\n", True))
            return {"messages": [], "delivery_context": ""}

        summary = state.get("summary", "")
        messages = state.get("messages", [])[-context.recent_messages :]
        opening_prompt = PromptTemplate.from_template(context.opening_prompt).format(
            conversation_summary=summary or "No previous conversation.",
            recent_conversation=self._format_messages(messages),
        )
        request = LanguageModelRequest(
            LanguageModelRole.FAST,
            self._request_messages(
                state,
                context,
                summary,
                [*messages, HumanMessage(content=opening_prompt)],
            ),
        )
        response = await self._stream_visible(request)
        return {"messages": [AIMessage(response)], "delivery_context": ""}

    async def reply(
        self,
        state: ConversationState,
        runtime: ConversationRuntime,
    ) -> dict[str, list[AnyMessage]]:
        context = runtime.context
        summary = state.get("summary", "")
        messages = state["messages"][-context.recent_messages :]
        user_text = next(
            (
                message.text
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        plan = self._response_router.plan(user_text)
        if context.classify_ambiguous and self._response_router.is_ambiguous(user_text):
            plan = await self._classify(user_text)
        role = (
            LanguageModelRole.DETAILED
            if plan.mode is ResponseMode.DETAILED
            else LanguageModelRole.FAST
        )
        request = LanguageModelRequest(
            role,
            self._request_messages(state, context, summary, messages),
        )
        response = await self._stream_visible(
            request,
            acknowledgement_delay=(
                context.acknowledgement_delay if plan.acknowledge else None
            ),
        )
        return {"messages": [AIMessage(response)]}

    async def _classify(self, text: str) -> ResponsePlan:
        classification = ""
        async for chunk in self._language_model.generate(
            LanguageModelRequest(
                LanguageModelRole.CLASSIFIER,
                (
                    ConversationMessage(
                        ConversationRole.SYSTEM,
                        self._response_router.CLASSIFICATION_PROMPT,
                    ),
                    ConversationMessage(ConversationRole.USER, text),
                ),
            )
        ):
            classification += chunk.content
        return self._response_router.classified_plan(classification)

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
        content = ""
        async for chunk in self._language_model.generate(
            LanguageModelRequest(
                LanguageModelRole.FAST,
                self._domain_messages(
                    [
                        *self._delivery_messages(state),
                        HumanMessage(content=summary_prompt),
                    ]
                ),
            )
        ):
            content += chunk.content
        return {"summary": content, "delivery_context": ""}

    async def _stream_visible(
        self,
        request: LanguageModelRequest,
        acknowledgement_delay: float | None = None,
    ) -> str:
        writer = get_stream_writer()
        content = ""
        stream = self._language_model.generate(request)
        if acknowledgement_delay is not None:
            first_chunk = asyncio.create_task(anext(stream))
            try:
                chunk = await asyncio.wait_for(
                    asyncio.shield(first_chunk), acknowledgement_delay
                )
            except TimeoutError:
                if self._profile_preparation is not None and (
                    reaction := self._profile_preparation.next_reaction()
                ):
                    writer(ConversationTextChunk(f"{reaction}\n", True))
                chunk = await first_chunk
            except StopAsyncIteration:
                return content
            except asyncio.CancelledError:
                first_chunk.cancel()
                with suppress(asyncio.CancelledError):
                    await first_chunk
                await stream.aclose()
                raise
            content += chunk.content
            writer(ConversationTextChunk(chunk.content))

        async for chunk in stream:
            content += chunk.content
            writer(ConversationTextChunk(chunk.content))
        return content

    @classmethod
    def _request_messages(
        cls,
        state: ConversationState,
        context: ConversationContext,
        summary: str,
        messages: list[AnyMessage],
    ) -> tuple[ConversationMessage, ...]:
        return cls._domain_messages(
            [
                SystemMessage(
                    content=PromptTemplate.from_template(context.system_prompt).format(
                        conversation_summary=(
                            "The current conversation summary is supplied in the "
                            "next system message."
                        )
                    )
                ),
                SystemMessage(
                    content="Conversation summary: "
                    f"{summary or 'No previous conversation.'}"
                ),
                *cls._delivery_messages(state),
                *messages,
            ]
        )

    @staticmethod
    def _domain_messages(
        messages: list[AnyMessage],
    ) -> tuple[ConversationMessage, ...]:
        roles = {
            "system": ConversationRole.SYSTEM,
            "human": ConversationRole.USER,
            "ai": ConversationRole.ASSISTANT,
        }
        system_content = "\n\n".join(
            message.text
            for message in messages
            if message.type == "system" and message.text
        )
        conversation = tuple(
            ConversationMessage(roles[message.type], message.text)
            for message in messages
            if message.type != "system" and message.text
        )
        if not system_content:
            return conversation
        return (
            ConversationMessage(ConversationRole.SYSTEM, system_content),
            *conversation,
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
        return [SystemMessage(content=context)] if context else []
