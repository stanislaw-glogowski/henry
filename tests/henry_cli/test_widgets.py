import pytest

from henry_cli.events import TelemetrySnapshot
from henry_cli.ui.state import PipelineState
from henry_cli.ui.widgets import (
    PIPELINE_STATUS_SEGMENT_WIDTH,
    SCORE_WIDTH,
    PipelinePanel,
    StatusBar,
    StatusModal,
    TelemetryPanel,
)
from henry_client.pipeline import PipelineStageStatus


def test_status_bar_builds_scaled_segments_and_optional_text() -> None:
    bar = StatusBar(segment_width=2)

    result = bar.add_active("green", 2)
    assert result is None
    bar.add_inactive(1)

    assert bar.finish("ready").plain == "[████░░] ready"


@pytest.mark.parametrize("status", list(PipelineStageStatus))
def test_pipeline_status_renders_three_equal_segments(
    status: PipelineStageStatus,
) -> None:
    bar = PipelinePanel._render_status(status)

    assert bar.plain.startswith("[")
    assert bar.plain.endswith("]")
    assert len(bar.plain) == PIPELINE_STATUS_SEGMENT_WIDTH * 3 + 2


def test_pipeline_panel_renders_all_stages() -> None:
    table = PipelinePanel().render()

    assert table.row_count == 7


@pytest.mark.parametrize("detected", [False, True])
def test_telemetry_bar_renders_score_and_detection_style(detected: bool) -> None:
    bar = TelemetryPanel._render_bar(0.5, detected)

    assert bar.plain.endswith("0.500")
    assert bar.plain.count("█") == round(0.5 * SCORE_WIDTH)
    assert bar.plain.count("░") == SCORE_WIDTH - round(0.5 * SCORE_WIDTH)


def test_telemetry_panel_renders_scores_and_sample_counts() -> None:
    panel = TelemetryPanel()
    panel.snapshot = TelemetrySnapshot(
        captured_sample_count=12_345,
        played_sample_count=6_789,
        speech_score=0.25,
        speech_detected=False,
        wakeword_score=0.75,
        wakeword_detected=True,
    )

    assert panel.render().row_count == 4


def test_unmounted_status_modal_skips_content_refresh() -> None:
    modal = StatusModal(PipelineState())

    modal.watch_state(PipelineState())

    assert not modal.is_mounted
