from dataclasses import dataclass

from henry import TelemetrySink
from henry.telemetry import (
    AudioFrameCaptured,
    AudioFramePlayed,
    TelemetryMeasurement,
    VadObserved,
)


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    vad_score: float = 0.0
    is_speech: bool = False
    captured_sample_count: int = 0
    played_sample_count: int = 0


class TelemetryCollector(TelemetrySink):
    def __init__(self):
        self._vad_score: float = 0.0
        self._is_speech: bool = False
        self._captured_sample_count: int = 0
        self._played_sample_count: int = 0

    def publish(self, *measurements: TelemetryMeasurement) -> None:
        for measurement in measurements:
            match measurement:
                case VadObserved() as vad:
                    self._vad_score = vad.score
                    self._is_speech = vad.is_speech
                case AudioFrameCaptured(sample_count):
                    self._captured_sample_count += sample_count
                case AudioFramePlayed(sample_count):
                    self._played_sample_count += sample_count

    def snapshot(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            vad_score=self._vad_score,
            is_speech=self._is_speech,
            captured_sample_count=self._captured_sample_count,
            played_sample_count=self._played_sample_count,
        )
