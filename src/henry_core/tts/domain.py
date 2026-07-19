from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranscriptionChunk:
    text: str
