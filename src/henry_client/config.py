from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VADConfig:
    threshold: float = 0.5


@dataclass(frozen=True, slots=True)
class WakeWordConfig:
    reply_message: str | None = None
    threshold: float = 0.75
    model_path: str | None = None
