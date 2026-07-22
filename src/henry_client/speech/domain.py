from collections import deque

from ..audio import AudioBuffer, AudioChunk, AudioFrame

MIN_START_SPEECH_FRAMES = 10
MAX_START_SILENCE_FRAMES = 150
MAX_END_SILENCE_FRAMES = 50
PRE_ROLL_FRAMES = 15


class SpeechSegmenter:
    def __init__(
        self,
        min_start_speech_frames: int = MIN_START_SPEECH_FRAMES,
        max_start_silence_frames: int = MAX_START_SILENCE_FRAMES,
        max_end_silence_frames: int = MAX_END_SILENCE_FRAMES,
        pre_roll_frames: int = PRE_ROLL_FRAMES,
    ) -> None:

        self._min_start_speech_frames = min_start_speech_frames
        self._max_start_silence_frames = max_start_silence_frames
        self._max_end_silence_frames = max_end_silence_frames

        self._chunks = AudioBuffer()
        self._pending_chunks: deque[AudioChunk] = deque(
            maxlen=pre_roll_frames + min_start_speech_frames
        )
        self._speech_started = False
        self._start_speech_frames = 0
        self._start_silence_frames = 0
        self._end_silence_frames = 0

    def feed(
        self,
        chunk: AudioChunk,
    ) -> tuple[bool, AudioFrame | None]:
        if not self._speech_started:
            self._pending_chunks.append(chunk)

            if chunk.is_speech:
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
                    self._reset()
                    return True, None

            return False, None

        self._chunks.append(chunk)

        if chunk.is_speech:
            self._end_silence_frames = 0
        else:
            self._end_silence_frames += 1
            if self._end_silence_frames > self._max_end_silence_frames:
                return True, self._build()

        return False, None

    def _build(self) -> AudioFrame | None:
        frame = self._chunks.build()
        self._reset()
        return frame

    def _reset(self) -> None:
        self._chunks.clear()
        self._pending_chunks.clear()
        self._speech_started = False
        self._start_speech_frames = 0
        self._start_silence_frames = 0
        self._end_silence_frames = 0
