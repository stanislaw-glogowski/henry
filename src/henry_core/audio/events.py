from dataclasses import dataclass

from ..events import AppEvent


@dataclass(frozen=True, slots=True)
class AudioFrameCaptured(AppEvent):
    samples_count: int


@dataclass(frozen=True, slots=True)
class AudioFramePlayed(AppEvent):
    samples_count: int
