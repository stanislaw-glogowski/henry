from itertools import cycle

from .config import ConversationReactions
from .domain import LanguageModelRole
from .model import LanguageModelService


class ProfilePreparation:
    """Rotate immediate reactions and warm the fast model outside response runs."""

    def __init__(
        self,
        language_model: LanguageModelService,
        reactions: ConversationReactions,
    ) -> None:
        self._language_model = language_model
        self._wake_reactions = cycle(reactions.wake)
        self._wait_reactions = cycle(reactions.wait)

    async def prepare(self) -> None:
        await self._language_model.prepare(LanguageModelRole.FAST)

    def next_wake_reaction(self) -> str | None:
        return next(self._wake_reactions, None)

    def next_reaction(self) -> str | None:
        return next(self._wait_reactions, None)
