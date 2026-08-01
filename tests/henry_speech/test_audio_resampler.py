import numpy as np
import pytest

from henry_speech.audio import AudioFormat, AudioResampler


def test_audio_resampler_preserves_stream_state_and_target_format() -> None:
    source = AudioFormat(sample_rate=44_100, channels=1)
    target = AudioFormat(sample_rate=16_000, channels=1)
    resampler = AudioResampler(target)

    first = resampler.process(
        source.build_frame(np.full(4_410, 0.25, dtype=np.float32))
    )
    second = resampler.process(
        source.build_frame(np.full(4_410, -0.25, dtype=np.float32))
    )

    assert first.format == target
    assert second.format == target
    assert first.samples.size > 0
    assert second.samples.size > 0


def test_audio_resampler_passes_through_matching_rate_and_can_reset() -> None:
    target = AudioFormat(sample_rate=16_000, channels=1)
    resampler = AudioResampler(target)
    samples = np.asarray([0.25, -0.5], dtype=np.float32)

    unchanged = resampler.process(target.build_frame(samples))
    np.testing.assert_array_equal(unchanged.samples, samples)

    resampler.reset()
    converted = resampler.process(
        AudioFormat(48_000, 1).build_frame(np.ones(4_800, dtype=np.float32))
    )
    assert converted.format == target
    assert converted.samples.size > 0


def test_audio_resampler_validates_formats_and_sample_alignment() -> None:
    with pytest.raises(ValueError, match="sample rate"):
        AudioResampler(AudioFormat(0, 1))
    with pytest.raises(ValueError, match="channel count"):
        AudioResampler(AudioFormat(16_000, 0))

    resampler = AudioResampler(AudioFormat(16_000, 1))
    with pytest.raises(RuntimeError, match="cannot change channel count"):
        resampler.process(
            AudioFormat(16_000, 2).build_frame(np.asarray([0.1, 0.2], dtype=np.float32))
        )

    stereo = AudioResampler(AudioFormat(16_000, 2))
    with pytest.raises(ValueError, match="not aligned"):
        stereo.process(
            AudioFormat(16_000, 2).build_frame(
                np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
            )
        )

    resampler.process(
        AudioFormat(44_100, 1).build_frame(np.ones(4_410, dtype=np.float32))
    )
    with pytest.raises(RuntimeError, match="sample rate mismatch"):
        resampler.process(
            AudioFormat(48_000, 1).build_frame(np.ones(4_800, dtype=np.float32))
        )
