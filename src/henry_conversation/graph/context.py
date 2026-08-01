from dataclasses import dataclass

from langgraph.runtime import Runtime

from ..config import ConversationProfile

type ConversationRuntime = Runtime[ConversationContext]


@dataclass(frozen=True, slots=True)
class ConversationContext:
    model: str
    recent_messages: int
    system_prompt: str
    opening_prompt: str
    summary_prompt: str

    @staticmethod
    def from_profile(
        profile: ConversationProfile,
    ) -> ConversationContext:
        return ConversationContext(
            model=profile.model,
            recent_messages=profile.recent_messages,
            system_prompt=profile.prompts.system,
            opening_prompt=profile.prompts.opening,
            summary_prompt=profile.prompts.summary,
        )
