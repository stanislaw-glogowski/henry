from henry_client.speech.domain import SpeechSegmenter
from tests.support import chunk


def test_segmenter_builds_utterance_after_speech_and_trailing_silence() -> None:
    segmenter = SpeechSegmenter()

    for _ in range(segmenter._MIN_START_SPEECH_FRAMES):
        ended, value = segmenter.feed(chunk(speech_detected=True, speech_score=0.9))
        assert not ended
        assert value is None

    for _ in range(segmenter._MAX_END_SILENCE_FRAMES):
        ended, value = segmenter.feed(chunk())
        assert not ended
        assert value is None

    ended, value = segmenter.feed(chunk())

    assert ended
    assert value is not None
    assert len(value.samples) > 0


def test_segmenter_reports_timeout_before_speech() -> None:
    segmenter = SpeechSegmenter()

    for _ in range(segmenter._MAX_START_SILENCE_FRAMES):
        assert segmenter.feed(chunk()) == (False, None)

    assert segmenter.feed(chunk()) == (True, None)


def test_segmenter_requires_consecutive_speech_frames() -> None:
    segmenter = SpeechSegmenter()

    for _ in range(segmenter._MIN_START_SPEECH_FRAMES - 1):
        segmenter.feed(chunk(speech_detected=True, speech_score=0.9))
    segmenter.feed(chunk())

    assert segmenter.feed(chunk(speech_detected=True, speech_score=0.9)) == (
        False,
        None,
    )
