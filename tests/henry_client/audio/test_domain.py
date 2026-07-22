import numpy as np
import pytest

from henry_client.audio import AudioBuffer, AudioFormat, AudioFormatError
from tests.support import frame


def test_audio_format_verifies_channels_and_sample_rate() -> None:
    expected = AudioFormat(sample_rate=16_000, channels=1)

    expected.verify(AudioFormat(sample_rate=16_000, channels=1))

    with pytest.raises(AudioFormatError, match="channel count"):
        expected.verify(AudioFormat(sample_rate=16_000, channels=2))
    with pytest.raises(AudioFormatError, match="sample rate"):
        expected.verify(AudioFormat(sample_rate=22_050, channels=1))


def test_audio_buffer_concatenates_frames_and_preserves_format() -> None:
    buffer = AudioBuffer()
    buffer.append(frame(1.0, samples_count=2))
    buffer.append(frame(2.0, samples_count=3))

    result = buffer.build()

    assert result is not None
    assert result.sample_rate == 16_000
    assert result.channels == 1
    np.testing.assert_array_equal(result.samples, [1.0, 1.0, 2.0, 2.0, 2.0])


def test_audio_buffer_rejects_incompatible_frames() -> None:
    buffer = AudioBuffer()
    buffer.append(frame(sample_rate=16_000))

    with pytest.raises(AudioFormatError):
        buffer.append(frame(sample_rate=22_050))


def test_empty_audio_buffer_builds_nothing() -> None:
    assert AudioBuffer().build() is None
