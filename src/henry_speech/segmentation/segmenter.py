from collections import deque

from ..audio import AudioBuffer, AudioFrame
from ..capture import SpeechChunk
from .config import SegmentationSettings
from .domain import SpeechSegment


class UtteranceSegmenter:
    """Assemble VAD-labelled frames into bounded utterances with pre-roll."""

    def __init__(
        self,
        settings: SegmentationSettings | None = None,
    ) -> None:
        if settings is None:
            settings = SegmentationSettings()

        self._settings = settings
        self._buffer = AudioBuffer()
        self._pending_frames: deque[AudioFrame] = deque(
            maxlen=settings.pre_roll_frames + settings.min_start_speech_frames
        )
        self._speech_started = False
        self._start_speech_frames = 0
        self._start_silence_frames = 0
        self._end_silence_frames = 0
        self._speech_frames = 0
        self._utterance_frames = 0

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
        self._speech_frames = 0
        self._utterance_frames = 0

    def _feed(
        self,
        chunk: SpeechChunk,
    ) -> bool:
        """Return completion status and an utterance, or ``None`` for timeout."""
        if not self._speech_started:
            self._pending_frames.append(chunk.audio)

            if chunk.is_speech:
                self._start_speech_frames += 1
                if self._start_speech_frames >= self._settings.min_start_speech_frames:
                    self._speech_started = True
                    self._speech_frames = self._start_speech_frames
                    self._utterance_frames = len(self._pending_frames)
                    self._start_silence_frames = 0
                    for pending_frame in self._pending_frames:
                        self._buffer.append(pending_frame)
                    self._pending_frames.clear()
            else:
                self._start_speech_frames = 0
                self._start_silence_frames += 1
                if self._start_silence_frames > self._settings.max_start_silence_frames:
                    self._start_silence_frames = 0
                    return True

            return False

        self._buffer.append(chunk.audio)
        self._utterance_frames += 1

        if chunk.is_speech:
            self._speech_frames += 1
            self._end_silence_frames = 0
        else:
            self._end_silence_frames += 1
            if self._end_silence_frames >= self._end_silence_limit:
                return True

        return self._utterance_frames >= self._settings.max_utterance_frames

    @property
    def _end_silence_limit(self) -> int:
        if self._speech_frames < self._settings.short_utterance_speech_frames:
            return self._settings.short_utterance_end_silence_frames
        return self._settings.max_end_silence_frames
