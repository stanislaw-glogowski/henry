import numpy as np
import pytest
from pydantic import ValidationError

from henry_speech.audio import AudioFormat
from henry_speech.audio.adapters.pyaudio import (
    PyAudioDriver,
    PyAudioInput,
    PyAudioOutput,
)
from henry_speech.capture import DetectionResult, SpeechChunk, WakeWordProfile
from henry_speech.config import SpeechProfile, SpeechSettings
from henry_speech.events import (
    InteractionTimingObserved,
    SpeechChunkCaptured,
    VADObserved,
    WakeWordObserved,
)
from henry_speech.segmentation import UtteranceSegmenter
from henry_speech.segmentation.config import SegmentationSettings
from henry_speech.synthesis.config import MLXChatterboxSettings
from henry_speech.transcription import TurnEndpointDetector
from henry_speech.transcription.config import MLXWhisperSettings

FORMAT = AudioFormat(sample_rate=16_000, channels=1)


def frame(*samples: float):
    return FORMAT.build_frame(np.asarray(samples, dtype=np.float32))


def chunk(*, speech: bool, wakeword: bool = False) -> SpeechChunk:
    return SpeechChunk(
        audio=frame(0.1, 0.2),
        vad=DetectionResult(score=0.8 if speech else 0.1, detected=speech),
        wakeword=DetectionResult(score=0.9, detected=wakeword),
    )


def test_audio_format_and_frames() -> None:
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


def test_pyaudio_adapter_package_exports_concrete_types() -> None:
    assert PyAudioDriver.__name__ == "PyAudioDriver"
    assert PyAudioInput.__name__ == "PyAudioInput"
    assert PyAudioOutput.__name__ == "PyAudioOutput"


def test_capture_domain_and_telemetry_events() -> None:
    speech = chunk(speech=True, wakeword=True)
    delayed_wakeword = chunk(speech=False, wakeword=True)
    silence = SpeechChunk(
        audio=frame(0.0),
        vad=DetectionResult(),
        wakeword=None,
    )
    assert speech.is_speech
    assert speech.is_wakeword
    assert not delayed_wakeword.is_speech
    assert delayed_wakeword.is_wakeword
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
    assert InteractionTimingObserved("turn_ready", 0).elapsed_ms == 0


def test_segmentation_detects_utterance_timeout_and_reset() -> None:
    settings = SegmentationSettings(
        min_start_speech_frames=2,
        max_start_silence_frames=1,
        max_end_silence_frames=2,
        short_utterance_speech_frames=1,
        short_utterance_end_silence_frames=2,
        max_utterance_frames=10,
        pre_roll_frames=1,
    )
    segmenter = UtteranceSegmenter(settings)

    assert segmenter.feed(chunk(speech=False)) == (False, None)
    assert segmenter.feed(chunk(speech=False)) == (True, None)
    assert segmenter.feed(chunk(speech=True)) == (False, None)
    assert segmenter.feed(chunk(speech=False)) == (False, None)
    assert segmenter.feed(chunk(speech=True)) == (False, None)
    assert segmenter.feed(chunk(speech=True)) == (False, None)
    assert segmenter.feed(chunk(speech=False)) == (False, None)
    ended, segment = segmenter.feed(chunk(speech=False))
    assert ended
    assert segment is not None
    assert segment.audio.samples_count == 10
    segmenter.reset()


def test_segmentation_uses_longer_pause_for_short_utterance_and_hard_limit() -> None:
    segmenter = UtteranceSegmenter(
        SegmentationSettings(
            min_start_speech_frames=1,
            max_start_silence_frames=2,
            max_end_silence_frames=1,
            short_utterance_speech_frames=3,
            short_utterance_end_silence_frames=2,
            max_utterance_frames=5,
            pre_roll_frames=0,
        )
    )

    assert segmenter.feed(chunk(speech=True)) == (False, None)
    assert segmenter.feed(chunk(speech=False)) == (False, None)
    ended, segment = segmenter.feed(chunk(speech=False))
    assert ended and segment is not None

    segmenter.feed(chunk(speech=True))
    segmenter.feed(chunk(speech=True))
    segmenter.feed(chunk(speech=True))
    ended, segment = segmenter.feed(chunk(speech=False))
    assert ended and segment is not None

    for _ in range(4):
        assert segmenter.feed(chunk(speech=True)) == (False, None)
    ended, segment = segmenter.feed(chunk(speech=True))
    assert ended and segment is not None


def test_speech_configuration_defaults_and_validation() -> None:
    profile = SpeechProfile(
        wakeword=WakeWordProfile(label="Wake", model_path="wake.onnx"),
        tts={"model_path": "voice.onnx"},
        stt={},
    )
    assert profile.tts_piper.model_path == "voice.onnx"
    assert profile.stt_mlx_parakeet_tdt.model_id is None
    assert SpeechSettings().audio.driver == "avfaudio"
    assert SpeechSettings().segmentation.min_start_speech_frames == 10

    alternate = SpeechSettings.model_validate(
        {
            "tts": {"adapter": "mlx:chatterbox"},
            "stt": {"adapter": "mlx:whisper"},
        }
    )
    assert isinstance(alternate.tts, MLXChatterboxSettings)
    assert isinstance(alternate.stt, MLXWhisperSettings)

    with pytest.raises(ValidationError, match="ONNX"):
        WakeWordProfile(label="Wake", model_path="wake.bin")
    with pytest.raises(ValidationError, match="label"):
        WakeWordProfile.model_validate({"model_path": "wake.onnx"})
    with pytest.raises(ValidationError, match="model_path"):
        WakeWordProfile.model_validate({"model": "wake.onnx"})

    with pytest.raises(ValidationError, match="extra_forbidden"):
        SpeechSettings.model_validate(
            {"stt": {"adapter": "mlx:parakeet-tdt", "language": "pl"}}
        )


def test_turn_endpoint_detector_recognizes_continuations() -> None:
    detector = TurnEndpointDetector()

    assert detector.is_complete("Jaka jest pogoda")
    assert detector.is_complete("To wszystko.")
    assert not detector.is_complete("Chcę wiedzieć, ponieważ")
    assert not detector.is_complete("Jeszcze jedna rzecz...")
    assert not detector.is_complete("")
    assert detector.is_complete("123")


def test_segmentation_configuration_rejects_inconsistent_limits() -> None:
    with pytest.raises(ValidationError, match="short_utterance_end_silence_frames"):
        SegmentationSettings(
            max_end_silence_frames=10,
            short_utterance_end_silence_frames=9,
        )
    with pytest.raises(ValidationError, match="max_utterance_frames"):
        SegmentationSettings(
            short_utterance_speech_frames=10,
            max_utterance_frames=10,
        )
