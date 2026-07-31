from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from .context import ReplyContext
from .node import ReplyNode
from .state import ReplyState


class ReplyGraph:
    NAME = "henry-reply"

    def __init__(
        self,
        node: ReplyNode,
        checkpointer: Checkpointer = None,
    ):
        # noinspection PyTypeChecker
        builder = StateGraph(ReplyState, ReplyContext)

        # noinspection PyTypeChecker
        builder.add_node(node.NAME, node)
        builder.add_edge(START, node.NAME)
        builder.add_edge(node.NAME, END)
        self._compiled = builder.compile(checkpointer=checkpointer)

    @property
    def compiled(self) -> CompiledStateGraph[StateGraph, ReplyContext, Any, Any]:
        return self._compiled
