from itertools import cycle
from random import shuffle

from ..model import LanguageModelRole, LanguageModelService
from .config import ConversationReactions


class ProfilePreparation:
    """Rotate shuffled reactions and warm the fast model outside response runs."""

    def __init__(
        self,
        language_model: LanguageModelService,
        reactions: ConversationReactions,
    ) -> None:
        self._language_model = language_model
        wake_reactions = list(reactions.wake)
        wait_reactions = list(reactions.wait)
        shuffle(wake_reactions)
        shuffle(wait_reactions)
        self._wake_reactions = cycle(wake_reactions)
        self._wait_reactions = cycle(wait_reactions)

    async def prepare(self) -> None:
        await self._language_model.prepare(LanguageModelRole.FAST)

    def next_wake_reaction(self) -> str | None:
        return next(self._wake_reactions, None)

    def next_reaction(self) -> str | None:
        return next(self._wait_reactions, None)
