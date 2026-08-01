import numpy as np
import pytest

from henry_speech.audio import AudioBuffer, AudioFormat


def test_audio_buffer_accumulates_matching_frames_and_clears() -> None:
    audio_format = AudioFormat(sample_rate=16_000, channels=1)
    buffer = AudioBuffer()

    assert len(buffer) == 0
    assert buffer.build() is None
    buffer.append(audio_format.build_frame(np.asarray([1.0], dtype=np.float32)))
    buffer.append(audio_format.build_frame(np.asarray([2.0], dtype=np.float32)))

    assert len(buffer) == 2
    np.testing.assert_array_equal(buffer.build().samples, [1.0, 2.0])

    with pytest.raises(RuntimeError, match="sample rate"):
        buffer.append(
            AudioFormat(8_000, 1).build_frame(np.asarray([0.0], dtype=np.float32))
        )

    buffer.clear()
    assert buffer.build() is None
