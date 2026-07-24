from collections import deque

from ..audio import AudioBuffer, AudioFrame


class SpeechSegmenter:
    """Build utterances from consecutive VAD-classified audio frames."""

    _MIN_START_SPEECH_FRAMES = 10
    _MAX_START_SILENCE_FRAMES = 150
    _MAX_END_SILENCE_FRAMES = 50
    _PRE_ROLL_FRAMES = 15

    def __init__(self) -> None:
        self._buffer = AudioBuffer()
        self._pending_frames: deque[AudioFrame] = deque(
            maxlen=self._PRE_ROLL_FRAMES + self._MIN_START_SPEECH_FRAMES
        )
        self._speech_started = False
        self._start_speech_frames = 0
        self._start_silence_frames = 0
        self._end_silence_frames = 0

    def feed(
        self,
        frame: AudioFrame,
        speech_detected: bool,
    ) -> tuple[bool, AudioFrame | None]:
        """Return completion status and an utterance, or ``None`` for timeout."""
        if not self._speech_started:
            self._pending_frames.append(frame)

            if speech_detected:
                self._start_speech_frames += 1
                if self._start_speech_frames >= self._MIN_START_SPEECH_FRAMES:
                    self._speech_started = True
                    self._start_silence_frames = 0
                    for pending_frame in self._pending_frames:
                        self._buffer.append(pending_frame)
                    self._pending_frames.clear()
            else:
                self._start_speech_frames = 0
                self._start_silence_frames += 1
                if self._start_silence_frames > self._MAX_START_SILENCE_FRAMES:
                    self._start_silence_frames = 0
                    self._reset()
                    return True, None

            return False, None

        self._buffer.append(frame)

        if speech_detected:
            self._end_silence_frames = 0
        else:
            self._end_silence_frames += 1
            if self._end_silence_frames > self._MAX_END_SILENCE_FRAMES:
                return True, self._build()

        return False, None

    def _build(self) -> AudioFrame | None:
        frame = self._buffer.build()
        self._reset()
        return frame

    def _reset(self) -> None:
        self._buffer.clear()
        self._pending_frames.clear()
        self._speech_started = False
        self._start_speech_frames = 0
        self._start_silence_frames = 0
        self._end_silence_frames = 0
