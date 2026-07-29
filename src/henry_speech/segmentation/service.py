from collections import deque
from dataclasses import dataclass

from ..audio import AudioBuffer, AudioFrame
from ..capture import SpeechChunk
from .domain import SpeechSegment


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    min_start_speech_frames: int = 10
    max_start_silence_frames: int = 150
    max_end_silence_frames: int = 50
    pre_roll_frames: int = 15


class SegmentationService:
    def __init__(
        self,
        config: SegmentationConfig,
    ) -> None:
        self._config = config
        self._buffer = AudioBuffer()
        self._pending_frames: deque[AudioFrame] = deque(
            maxlen=config.pre_roll_frames + config.min_start_speech_frames
        )
        self._speech_started = False
        self._start_speech_frames = 0
        self._start_silence_frames = 0
        self._end_silence_frames = 0

    def feed(
        self,
        chunk: SpeechChunk,
    ) -> tuple[bool, SpeechSegment | None]:
        ended = self._feed(chunk)
        if not ended:
            return False, None

        audio = self._buffer.build()
        self.reset()

        if audio is None:
            return True, None

        return True, SpeechSegment(
            audio=audio,
        )

    def reset(self) -> None:
        self._buffer.clear()
        self._pending_frames.clear()
        self._speech_started = False
        self._start_speech_frames = 0
        self._start_silence_frames = 0
        self._end_silence_frames = 0

    def _feed(
        self,
        chunk: SpeechChunk,
    ) -> bool:
        """Return completion status and an utterance, or ``None`` for timeout."""
        if not self._speech_started:
            self._pending_frames.append(chunk.audio)

            if chunk.voice_detected:
                self._start_speech_frames += 1
                if self._start_speech_frames >= self._config.min_start_speech_frames:
                    self._speech_started = True
                    self._start_silence_frames = 0
                    for pending_frame in self._pending_frames:
                        self._buffer.append(pending_frame)
                    self._pending_frames.clear()
            else:
                self._start_speech_frames = 0
                self._start_silence_frames += 1
                if self._start_silence_frames > self._config.max_start_silence_frames:
                    self._start_silence_frames = 0
                    return True

            return False

        self._buffer.append(chunk.audio)

        if chunk.voice_detected:
            self._end_silence_frames = 0
        else:
            self._end_silence_frames += 1
            if self._end_silence_frames > self._config.max_end_silence_frames:
                return True

        return False
