from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Footer, RichLog

from ..events import EventBridge
from ..logs import LogBuffer
from ..telemetry import TelemetryCollector
from .state import State
from .widgets import PipelinePanel, StatusModal, TelemetryPanel

VIEW_BINDINGS = Binding.Group(
    "View",
    compact=False,
)


class Layout(App[None]):
    state = reactive(State())

    CSS_PATH = "styles.tcss"

    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding(
            "p",
            "toggle_pipelines",
            description="Toggle pipelines panel",
            group=VIEW_BINDINGS,
        ),
        Binding(
            "t",
            "toggle_telemetry",
            description="Toggle telemetry panel",
            group=VIEW_BINDINGS,
        ),
        Binding(
            "l",
            "toggle_logs",
            description="Toggle logs panel",
            group=VIEW_BINDINGS,
        ),
        Binding("s", "show_status", "Status"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        logs: LogBuffer,
        events: EventBridge,
        telemetry: TelemetryCollector,
    ) -> None:
        super().__init__()
        self._logs = logs
        self._events = events
        self._telemetry = telemetry

    def compose(self) -> ComposeResult:
        yield PipelinePanel(id="pipeline")
        yield TelemetryPanel(id="telemetry")
        yield RichLog(id="logs", highlight=True, markup=False)
        yield Footer(show_command_palette=False)

    def action_toggle_pipelines(self) -> None:
        logs = self.query_one("#pipeline", PipelinePanel)
        logs.display = not logs.display

    def action_toggle_telemetry(self) -> None:
        logs = self.query_one("#telemetry", TelemetryPanel)
        logs.display = not logs.display

    def action_toggle_logs(self) -> None:
        logs = self.query_one("#logs", RichLog)
        logs.display = not logs.display

    def action_show_status(self) -> None:
        self.push_screen(
            StatusModal(self.state.pipeline),
        )

    def watch_state(self, state: State) -> None:
        self.query_one("#pipeline", PipelinePanel).state = state.pipeline
        self.query_one("#telemetry", TelemetryPanel).snapshot = state.telemetry

        if isinstance(self.screen, StatusModal):
            self.screen.state = state.pipeline

    @work(group="events", exclusive=True)
    async def consume_events(self) -> None:
        while True:
            batch = await self._events.receive()
            self.state = self.state.reduce_events(batch)

    def flush_logs(self) -> None:
        widget = self.query_one("#logs", RichLog)

        for line in self._logs.drain():
            widget.write(line)

    def refresh_telemetry(self) -> None:
        snapshot = self._telemetry.snapshot()
        self.state = self.state.replace_telemetry(snapshot)

    def on_mount(self) -> None:
        self.consume_events()
        self.set_interval(0.1, self.refresh_telemetry)
        self.set_interval(0.1, self.flush_logs)

    def on_unmount(self) -> None:
        pass
