from collections.abc import Iterator
from typing import Any

from ...config import LanguageModelProfile, LanguageModelsProfile
from ...domain import (
    ConversationRole,
    LanguageModelChunk,
    LanguageModelRequest,
    LanguageModelRole,
)
from ..ports import LanguageModel


class LangChainLanguageModel(LanguageModel):
    def __init__(self, profiles: LanguageModelsProfile) -> None:
        super().__init__()
        self._profiles = profiles
        self._models: dict[LanguageModelRole, Any] = {}
        self._init_chat_model: Any = None

    def open(self) -> None:
        if self._init_chat_model is not None:
            raise RuntimeError("LangChain language model adapter is already open")
        from langchain.chat_models import init_chat_model

        self._init_chat_model = init_chat_model
        self._logger.debug("Adapter OPENED")

    def close(self) -> None:
        self._models.clear()
        self._init_chat_model = None
        self._logger.debug("Adapter CLOSED")

    def prepare(self, role: LanguageModelRole) -> None:
        self._model(role)

    def generate(self, request: LanguageModelRequest) -> Iterator[LanguageModelChunk]:
        from langchain.messages import AIMessage, HumanMessage, SystemMessage

        message_types = {
            ConversationRole.SYSTEM: SystemMessage,
            ConversationRole.USER: HumanMessage,
            ConversationRole.ASSISTANT: AIMessage,
        }
        messages = [
            message_types[message.role](content=message.content)
            for message in request.messages
        ]
        for chunk in self._model(request.role).stream(messages):
            if chunk.text:
                yield LanguageModelChunk(chunk.text)

    def _model(self, role: LanguageModelRole) -> Any:
        if model := self._models.get(role):
            return model

        profile = self._profile(role)
        model_id = profile.model_for("langchain")
        if self._init_chat_model is None:
            raise RuntimeError("LangChain language model adapter is not open")
        model = self._init_chat_model(
            model_id,
            temperature=profile.temperature,
            top_p=profile.top_p,
            num_predict=profile.max_tokens,
            reasoning=self._reasoning(model_id, profile.thinking),
            base_url="http://localhost:11434",
        )
        self._models[role] = model
        self._logger.debug("Model LOADED: role='{}', model='{}'", role, model_id)
        return model

    @staticmethod
    def _reasoning(model_id: str, thinking: bool) -> bool | str:
        # GPT-OSS does not support disabling reasoning; low is its shortest mode.
        if not thinking and model_id.startswith("ollama:gpt-oss"):
            return "low"
        return thinking

    def _profile(self, role: LanguageModelRole) -> LanguageModelProfile:
        profile = getattr(self._profiles, role.value)
        if profile is None:
            raise RuntimeError(f"Language model role is not configured: {role!r}")
        return profile
