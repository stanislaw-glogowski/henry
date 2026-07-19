from dataclasses import dataclass

type TelemetryMeasurement = AudioFrameCaptured | AudioFramePlayed | VadObserved


@dataclass(frozen=True)
class VadObserved:
    score: float
    is_speech: bool


@dataclass(frozen=True)
class AudioFrameCaptured:
    sample_count: int


@dataclass(frozen=True)
class AudioFramePlayed:
    sample_count: int
