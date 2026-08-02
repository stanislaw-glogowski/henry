from dataclasses import dataclass

from ..audio import AudioFrame


@dataclass(frozen=True, slots=True)
class DetectionResult:
    score: float = 0.0
    detected: bool = False


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    audio: AudioFrame
    vad: DetectionResult
    wakeword: DetectionResult | None

    @property
    def is_speech(self) -> bool:
        return self.vad.detected

    @property
    def is_wakeword(self) -> bool:
        return self.wakeword is not None and self.wakeword.detected
