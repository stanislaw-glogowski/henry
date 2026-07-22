from typing import Self

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Label, LoadingIndicator, Static

from henry_client.events import PipelineStage, PipelineStageStatus

from ..events import TelemetrySnapshot
from .state import PipelineState

TABLE_LABEL_WIDTH = 22

SCORE_WIDTH = 27
SCORE_DEFAULT_STYLE = "cyan"
SCORE_DETECTED_STYLE = "green"

PIPELINE_STATUS_SEGMENT_WIDTH = 9
PIPELINE_STATUS_STYLES = {
    PipelineStageStatus.UNKNOWN: "white",
    PipelineStageStatus.READY: "blue",
    PipelineStageStatus.STARTED: "green",
    PipelineStageStatus.COMPLETED: "cyan",
    PipelineStageStatus.FAILED: "red",
}

STATUS_BAR_LEFT_CAP = "["
STATUS_BAR_RIGHT_CAP = "]"
STATUS_BAR_ACTIVE_CAP = "█"
STATUS_BAR_INACTIVE_CAP = "░"
STATUS_BAR_INACTIVE_STYLE = "zincs"


class StatusBar(Text):
    def __init__(self, segment_width=1):
        super().__init__(STATUS_BAR_LEFT_CAP)
        self._segment_width = segment_width

    def add_active(self, style: str, segment_width=1) -> None:
        self.append(
            STATUS_BAR_ACTIVE_CAP * (segment_width * self._segment_width), style=style
        )

    def add_inactive(self, segment_width=1) -> None:
        self.append(
            STATUS_BAR_INACTIVE_CAP * (segment_width * self._segment_width),
            style=STATUS_BAR_INACTIVE_STYLE,
        )

    def finish(self, text: str | None = None) -> Self:
        self.append(STATUS_BAR_RIGHT_CAP)

        if text is not None:
            self.append(" ")
            self.append(text)

        return self


class PipelinePanel(Static):
    state: PipelineState = reactive(PipelineState())

    def render(self) -> Table:
        table = Table.grid(expand=False, padding=(0, 1))
        table.add_column(style="bold cyan", width=TABLE_LABEL_WIDTH, justify="right")
        table.add_column(justify="left")

        for stage, status in self.state.stages.items():
            table.add_row(
                stage.label,
                self._render_status(status),
            )

        return table

    @staticmethod
    def _render_status(status: PipelineStageStatus) -> StatusBar:
        bar = StatusBar(PIPELINE_STATUS_SEGMENT_WIDTH)

        match status:
            case (
                PipelineStageStatus.UNKNOWN
                | PipelineStageStatus.READY
                | PipelineStageStatus.COMPLETED
            ):
                bar.add_active(style=PIPELINE_STATUS_STYLES[status])
                bar.add_inactive()
                bar.add_inactive()
            case PipelineStageStatus.STARTED:
                bar.add_inactive()
                bar.add_active(style=PIPELINE_STATUS_STYLES[status])
                bar.add_inactive()
            case PipelineStageStatus.FAILED:
                bar.add_inactive()
                bar.add_inactive()
                bar.add_active(style=PIPELINE_STATUS_STYLES[status])

        return bar.finish()


class TelemetryPanel(Static):
    snapshot: TelemetrySnapshot = reactive(TelemetrySnapshot())

    def render(self) -> Table:
        table = Table.grid(expand=False, padding=(0, 1))
        table.add_column(style="bold cyan", width=TABLE_LABEL_WIDTH, justify="right")
        table.add_column(justify="left")
        table.add_row(
            "Voice Activity",
            self._render_bar(self.snapshot.speech_score, self.snapshot.speech_detected),
        )
        table.add_row(
            "Wake Word",
            self._render_bar(
                self.snapshot.wakeword_score, self.snapshot.wakeword_detected
            ),
        )
        table.add_row(
            "Captured Samples",
            f"{self.snapshot.captured_sample_count:,}",
        )
        table.add_row(
            "Played Samples",
            f"{self.snapshot.played_sample_count:,}",
        )
        return table

    @staticmethod
    def _render_bar(score: float, detected: bool) -> Text:
        bar = StatusBar()

        active_width = round(score * SCORE_WIDTH)

        if detected:
            bar.add_active(
                segment_width=active_width,
                style=SCORE_DETECTED_STYLE,
            )
        else:
            bar.add_active(
                segment_width=active_width,
                style=SCORE_DEFAULT_STYLE,
            )

        bar.add_inactive(SCORE_WIDTH - active_width)

        return bar.finish(f"{score:.3f}")


class StatusModal(ModalScreen[None]):
    state: PipelineState = reactive(PipelineState())

    BINDINGS = [
        Binding("escape", "close", show=False),
        Binding("s", "close", show=False),
    ]

    def __init__(self, state: PipelineState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        with Vertical(id="status-container"):
            yield Label(id="status-label")
            yield LoadingIndicator()

    def action_close(self) -> None:
        self.dismiss()

    def watch_state(self, state: PipelineState) -> None:
        if self.is_mounted:
            self._refresh_content()

    def on_mount(self) -> None:
        self._refresh_content()

    def _refresh_content(self) -> None:
        container = self.query_one("#status-container", Vertical)
        label = self.query_one("#status-label", Label)

        match self.state.stages[PipelineStage.RECORDING]:
            case PipelineStageStatus.STARTED:
                container.add_class("recording")
                label.update("Recording")
            case _:
                container.remove_class("recording")
                label.update("Please wait")
