from dataclasses import dataclass

from ..events import AppEvent


@dataclass(frozen=True, slots=True)
class VADObserved(AppEvent):
    score: float
    is_speech: bool
