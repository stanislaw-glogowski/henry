from henry_client.speech.domain import SpeechSegmenter
from tests.support import frame


def test_segmenter_builds_utterance_after_speech_and_trailing_silence() -> None:
    segmenter = SpeechSegmenter()

    for _ in range(segmenter._MIN_START_SPEECH_FRAMES):
        ended, value = segmenter.feed(frame(), True)
        assert not ended
        assert value is None

    for _ in range(segmenter._MAX_END_SILENCE_FRAMES):
        ended, value = segmenter.feed(frame(), False)
        assert not ended
        assert value is None

    ended, value = segmenter.feed(frame(), False)

    assert ended
    assert value is not None
    assert len(value.samples) > 0


def test_segmenter_reports_timeout_before_speech() -> None:
    segmenter = SpeechSegmenter()

    for _ in range(segmenter._MAX_START_SILENCE_FRAMES):
        assert segmenter.feed(frame(), False) == (False, None)

    assert segmenter.feed(frame(), False) == (True, None)


def test_segmenter_requires_consecutive_speech_frames() -> None:
    segmenter = SpeechSegmenter()

    for _ in range(segmenter._MIN_START_SPEECH_FRAMES - 1):
        segmenter.feed(frame(), True)
    segmenter.feed(frame(), False)

    assert segmenter.feed(frame(), True) == (
        False,
        None,
    )
