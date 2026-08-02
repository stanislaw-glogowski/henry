from dataclasses import dataclass

from langgraph.runtime import Runtime

from ..config import ConversationProfile, ConversationSettings

type ConversationRuntime = Runtime[ConversationContext]


@dataclass(frozen=True, slots=True)
class ConversationContext:
    recent_messages: int
    system_prompt: str
    opening_prompt: str
    summary_prompt: str
    acknowledgement_delay: float
    classify_ambiguous: bool

    @staticmethod
    def from_profile(
        profile: ConversationProfile,
        settings: ConversationSettings | None = None,
    ) -> ConversationContext:
        settings = settings or ConversationSettings()
        return ConversationContext(
            recent_messages=profile.recent_messages,
            system_prompt=profile.prompts.system,
            opening_prompt=profile.prompts.opening,
            summary_prompt=profile.prompts.summary,
            acknowledgement_delay=settings.acknowledgement_delay,
            classify_ambiguous=(
                settings.classify_ambiguous and profile.models.classifier is not None
            ),
        )
