from collections.abc import Awaitable, Callable

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate

from .context import ReplyContext, ReplyRuntime
from .state import ReplyState


class ReplyNode(
    Callable[
        [ReplyState, ReplyRuntime],
        Awaitable[dict[str, list[AnyMessage]]],
    ]
):
    NAME = "reply_node"

    def __init__(self) -> None:
        self._model: BaseChatModel | None = None
        self._system_prompt: str | None = None

    async def __call__(
        self,
        state: ReplyState,
        runtime: ReplyRuntime,
    ) -> ReplyState:
        model, system_prompt = self._use_context(runtime.context)
        response = await model.ainvoke(
            [
                SystemMessage(content=system_prompt),
                *state["messages"],
            ]
        )

        return {
            "messages": [response],
        }

    def _use_context(self, context: ReplyContext) -> tuple[BaseChatModel, str]:
        if self._model is None or self._system_prompt is None:
            self._model = init_chat_model(
                context.model,
                temperature=0,
                base_url="http://localhost:11434",
            )
            self._system_prompt = PromptTemplate.from_template(
                context.system_prompt
            ).format(
                name=context.name,
                language=context.language,
            )

        assert self._model is not None
        assert self._system_prompt is not None

        return self._model, self._system_prompt
