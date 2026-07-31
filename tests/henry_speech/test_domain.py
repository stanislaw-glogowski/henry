import numpy as np
import pytest
from pydantic import ValidationError

from henry_speech.audio import AudioBuffer, AudioFormat
from henry_speech.capture import DetectionResult, SpeechChunk, WakeWordProfile
from henry_speech.config import SpeechProfile, SpeechSettings
from henry_speech.events import SpeechChunkCaptured, VADObserved, WakeWordObserved
from henry_speech.segmentation import SegmentationService
from henry_speech.segmentation.config import SegmentationSettings
from henry_speech.synthesis import TTSProfile
from henry_speech.transcription import STTProfile

FORMAT = AudioFormat(sample_rate=16_000, channels=1)


def frame(*samples: float):
    return FORMAT.build_frame(np.asarray(samples, dtype=np.float32))


def chunk(*, speech: bool, wakeword: bool = False) -> SpeechChunk:
    return SpeechChunk(
        audio=frame(0.1, 0.2),
        vad=DetectionResult(score=0.8 if speech else 0.1, detected=speech),
        wakeword=DetectionResult(score=0.9, detected=wakeword),
    )


def test_audio_format_frames_and_buffer() -> None:
    other_channels = AudioFormat(sample_rate=16_000, channels=2)
    other_rate = AudioFormat(sample_rate=8_000, channels=1)
    with pytest.raises(RuntimeError, match="channel"):
        FORMAT.verify(other_channels)
    with pytest.raises(RuntimeError, match="sample rate"):
        FORMAT.verify(other_rate)

    original = frame(0.25, -0.5)
    rebuilt = FORMAT.build_frame(original.to_bytes())
    assert rebuilt.samples_count == 2
    np.testing.assert_array_equal(rebuilt.samples, original.samples)

    buffer = AudioBuffer()
    assert len(buffer) == 0
    assert buffer.build() is None
    buffer.append(frame(1.0))
    buffer.append(frame(2.0))
    assert len(buffer) == 2
    np.testing.assert_array_equal(buffer.build().samples, [1.0, 2.0])
    with pytest.raises(RuntimeError, match="sample rate"):
        buffer.append(other_rate.build_frame(np.asarray([0.0], dtype=np.float32)))
    buffer.clear()
    assert buffer.build() is None


def test_capture_domain_and_telemetry_events() -> None:
    speech = chunk(speech=True, wakeword=True)
    silence = SpeechChunk(
        audio=frame(0.0),
        vad=DetectionResult(),
        wakeword=None,
    )
    assert speech.is_speech
    assert speech.is_wakeword
    assert not silence.is_speech
    assert not silence.is_wakeword
    assert VADObserved.from_chunk(speech) == VADObserved(score=0.8, detected=True)
    assert WakeWordObserved.from_chunk(silence) is None
    assert WakeWordObserved.from_chunk(speech) == WakeWordObserved(
        score=0.9, detected=True
    )
    assert SpeechChunkCaptured.from_chunk(speech) == SpeechChunkCaptured(
        samples_len=2,
        is_speech=True,
        is_wakeword=True,
    )


def test_segmentation_detects_utterance_timeout_and_reset() -> None:
    settings = SegmentationSettings(
        min_start_speech_frames=2,
        max_start_silence_frames=1,
        max_end_silence_frames=1,
        pre_roll_frames=1,
    )
    service = SegmentationService(settings)

    assert service.feed(chunk(speech=False)) == (False, None)
    assert service.feed(chunk(speech=False)) == (True, None)
    assert service.feed(chunk(speech=True)) == (False, None)
    assert service.feed(chunk(speech=False)) == (False, None)
    assert service.feed(chunk(speech=True)) == (False, None)
    assert service.feed(chunk(speech=True)) == (False, None)
    assert service.feed(chunk(speech=False)) == (False, None)
    ended, segment = service.feed(chunk(speech=False))
    assert ended
    assert segment is not None
    assert segment.audio.samples_count == 10
    service.reset()


def test_speech_configuration_defaults_and_validation() -> None:
    profile = SpeechProfile(
        wakeword=WakeWordProfile(model="wake.onnx"),
        tts=TTSProfile(model="voice.onnx"),
        stt=STTProfile(),
    )
    assert profile.stt.model is None
    assert SpeechSettings().audio.driver == "pyaudio"
    assert SpeechSettings().segmentation.min_start_speech_frames == 10

    with pytest.raises(ValidationError, match="ONNX"):
        WakeWordProfile(model="wake.bin")
