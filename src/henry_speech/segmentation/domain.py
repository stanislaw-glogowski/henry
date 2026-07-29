from dataclasses import dataclass

from ..audio import AudioFrame


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    audio: AudioFrame
