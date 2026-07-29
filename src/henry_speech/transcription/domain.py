from dataclasses import dataclass

type Transcription = TranscriptionChunk | TranscriptionText | None


@dataclass(frozen=True, slots=True)
class TranscriptionChunk:
    content: str


@dataclass(frozen=True, slots=True)
class TranscriptionText:
    content: str
