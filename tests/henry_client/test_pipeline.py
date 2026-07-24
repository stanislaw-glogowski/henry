import pytest

from henry_client.pipeline import PipelineStage


@pytest.mark.parametrize(
    ("stage", "label"),
    [
        (PipelineStage.CAPTURE, "Capturing"),
        (PipelineStage.LISTENING, "Listening"),
        (PipelineStage.RECORDING, "Recording"),
        (PipelineStage.TRANSCRIPTION, "Transcribing"),
        (PipelineStage.PROCESSING, "Processing"),
        (PipelineStage.SYNTHESIS, "Synthesising"),
        (PipelineStage.PLAYBACK, "Playback"),
    ],
)
def test_pipeline_stage_has_presentation_label(
    stage: PipelineStage,
    label: str,
) -> None:
    assert stage.label == label
