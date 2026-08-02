from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from .context import ConversationContext
from .nodes import ConversationNodes
from .state import ConversationInputKind, ConversationState


class ConversationGraph:
    NAME = "henry-conversation"

    def __init__(
        self,
        nodes: ConversationNodes,
        checkpointer: Checkpointer = None,
    ) -> None:
        # noinspection PyTypeChecker
        builder = StateGraph[Any, ConversationContext, Any, Any](
            ConversationState,
            ConversationContext,
        )

        # noinspection PyTypeChecker
        builder.add_node(nodes.OPENING, nodes.opening)
        builder.add_node(nodes.REPLY, nodes.reply)
        builder.add_node(nodes.SUMMARIZE, nodes.summarize)
        builder.add_conditional_edges(
            START,
            self._route,
            {
                "activation": nodes.OPENING,
                "user_turn": nodes.REPLY,
            },
        )
        builder.add_edge(nodes.OPENING, END)
        builder.add_edge(nodes.REPLY, nodes.SUMMARIZE)
        builder.add_edge(nodes.SUMMARIZE, END)
        self._compiled = builder.compile(checkpointer=checkpointer)

    @staticmethod
    def _route(state: ConversationState) -> ConversationInputKind:
        return state["input_kind"]

    @property
    def compiled(
        self,
    ) -> CompiledStateGraph[Any, ConversationContext, Any, Any]:
        return self._compiled
