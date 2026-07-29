from dataclasses import dataclass

from ..audio import AudioFrame


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    audio: AudioFrame
    voice_detected: bool
    voice_score: float
    wakeword_detected: bool | None
    wakeword_score: float | None
