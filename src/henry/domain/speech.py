from collections import deque
from dataclasses import dataclass

from .audio import AudioBuffer, AudioFrame


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    audio: AudioFrame
    speech_score: float
    is_speech: bool


@dataclass(slots=True)
class SpeechSegment:
    audio: AudioFrame


@dataclass(frozen=True, slots=True)
class SpeechTranscription:
    text: str


class SpeechSegmenter:
    def __init__(
        self,
        min_start_speech_frames: int = 10,
        max_start_silence_frames: int = 150,
        max_end_silence_frames: int = 50,
        pre_roll_frames: int = 15,
    ) -> None:

        self._min_start_speech_frames = min_start_speech_frames
        self._max_start_silence_frames = max_start_silence_frames
        self._max_end_silence_frames = max_end_silence_frames

        self._chunks = AudioBuffer()
        self._pending_chunks: deque[AudioFrame] = deque(
            maxlen=pre_roll_frames + min_start_speech_frames
        )
        self._speech_started = False
        self._start_speech_frames = 0
        self._start_silence_frames = 0
        self._end_silence_frames = 0

    def feed(
        self,
        frame: SpeechChunk,
    ) -> tuple[bool, SpeechSegment | None]:
        if not self._speech_started:
            self._pending_chunks.append(frame.audio)

            if frame.is_speech:
                self._start_speech_frames += 1
                if self._start_speech_frames >= self._min_start_speech_frames:
                    self._speech_started = True
                    self._start_silence_frames = 0
                    for pending_frame in self._pending_chunks:
                        self._chunks.append(pending_frame)
                    self._pending_chunks.clear()
            else:
                self._start_speech_frames = 0
                self._start_silence_frames += 1
                if self._start_silence_frames > self._max_start_silence_frames:
                    self._start_silence_frames = 0
                    return True, None

            return False, None

        self._chunks.append(frame.audio)

        if frame.is_speech:
            self._end_silence_frames = 0
        else:
            self._end_silence_frames += 1
            if self._end_silence_frames > self._max_end_silence_frames:
                return True, self._build()

        return False, None

    def _build(self) -> SpeechSegment | None:
        audio = self._chunks.build()
        if audio is None:
            return None

        self._reset()
        return SpeechSegment(
            audio=audio,
        )

    def _reset(self) -> None:
        self._chunks.clear()
        self._pending_chunks.clear()
        self._speech_started = False
        self._start_speech_frames = 0
        self._start_silence_frames = 0
        self._end_silence_frames = 0
