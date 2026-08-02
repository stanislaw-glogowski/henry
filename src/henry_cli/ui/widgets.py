from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, RichLog, Static

from ..progress import ProgressItem, ProgressSnapshot, ProgressStatus
from .state import (
    AssistantMessage,
    ConversationState,
    PhraseState,
    RuntimeMode,
    State,
    UserMessage,
)

_SPINNER_FRAMES = (
    "\u280b",
    "\u2819",
    "\u2839",
    "\u2838",
    "\u283c",
    "\u2834",
    "\u2826",
    "\u2827",
    "\u2807",
    "\u280f",
)

_PHRASE_STYLES: dict[PhraseState, str] = {
    PhraseState.QUEUED: "bold #42d3c7",
    PhraseState.SPEAKING: "bold #ffd166",
    PhraseState.DELIVERED: "#a7b0c0",
}

_MODE_STYLES: dict[RuntimeMode, str] = {
    RuntimeMode.STARTING: "bold #b69cff",
    RuntimeMode.WAITING: "bold #42d3c7",
    RuntimeMode.LISTENING: "bold #4cc9f0",
    RuntimeMode.TRANSCRIBING: "bold #f72585",
    RuntimeMode.THINKING: "bold #b69cff",
    RuntimeMode.SPEAKING: "bold #ffd166",
    RuntimeMode.SHUTTING_DOWN: "bold #ff6b7a",
}


def _meter(score: float, detected: bool, *, width: int = 18) -> Text:
    score = min(1.0, max(0.0, score))
    active = round(score * width)
    color = "#67e8a5" if detected else "#42a5f5"
    meter = Text()
    meter.append("\u2588" * active, style=color)
    meter.append("\u2591" * (width - active), style="#30384b")
    meter.append(f"  {score:0.3f}", style="#8b95a7")
    return meter


def _short(value: str, limit: int = 34) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}\u2026"


class HeaderBar(Static):
    state = reactive(State())

    def render(self) -> Text:
        title = Text(" H ", style="bold #08111f on #42d3c7")
        title.append("  HENRY", style="bold #eef4ff")
        title.append("  /  local voice intelligence", style="#64748b")
        title.append("\n")
        title.append(" \u25cf ", style=_MODE_STYLES[self.state.mode])
        title.append(self.state.mode.value, style=_MODE_STYLES[self.state.mode])
        title.append("  \u00b7  ", style="#465064")
        title.append(self.state.info.profile_name, style="#b6c2d6")
        return title


class InfoPanel(Static):
    state = reactive(State())

    def render(self) -> Group:
        state = self.state
        info = state.info
        telemetry = state.telemetry

        status = Table.grid(expand=True)
        status.add_column(width=12, style="#718096")
        status.add_column(ratio=1)
        status.add_row("VAD", _meter(telemetry.vad_score, telemetry.vad_detected))
        status.add_row(
            "WAKE WORD",
            _meter(telemetry.wakeword_score, telemetry.wakeword_detected),
        )
        status.add_row(
            "CAPTURED",
            Text(f"{telemetry.captured_sample_count:,} samples", style="#b6c2d6"),
        )

        runtime = Table.grid(expand=True, padding=(0, 1))
        runtime.add_column(width=10, style="#64748b")
        runtime.add_column(ratio=1, overflow="fold")
        runtime.add_row("PROFILE", Text(info.profile_name, style="bold #eef4ff"))
        runtime.add_row("AUDIO", Text(info.audio_driver, style="#42d3c7"))
        runtime.add_row("INPUT", Text(_short(info.input_device), style="#b6c2d6"))
        runtime.add_row("OUTPUT", Text(_short(info.output_device), style="#b6c2d6"))
        runtime.add_row(
            "VAD",
            Text(f"{info.vad_adapter}  /  {info.vad_threshold:.2f}", style="#b6c2d6"),
        )
        runtime.add_row(
            "WAKE",
            Text(
                f"{_short(info.wakeword_model, 23)}  /  {info.wakeword_threshold:.2f}",
                style="#b6c2d6",
            ),
        )
        runtime.add_row(
            "STT",
            Text(f"{info.stt_adapter}\n{_short(info.stt_model)}", style="#b6c2d6"),
        )
        runtime.add_row(
            "LLM",
            Text(f"{info.llm_adapter}\n{_short(info.llm_model)}", style="#b6c2d6"),
        )
        runtime.add_row(
            "TTS",
            Text(f"{info.tts_adapter}\n{_short(info.tts_model)}", style="#b6c2d6"),
        )

        timing = Table.grid(expand=True, padding=(0, 1))
        timing.add_column(ratio=1, style="#718096")
        timing.add_column(width=10, justify="right", style="#b69cff")
        if telemetry.timings:
            for stage, elapsed_ms in telemetry.timings:
                timing.add_row(stage.replace("_", " ").upper(), f"{elapsed_ms:,.0f} ms")
        else:
            timing.add_row("NO INTERACTION YET", "\u2014")

        return Group(
            Text("SIGNALS", style="bold #42d3c7"),
            status,
            Text("\nRUNTIME", style="bold #b69cff"),
            runtime,
            Text("\nLATENCY", style="bold #ffd166"),
            timing,
        )


class ConversationTranscript(Static):
    conversation = reactive(ConversationState())
    spinner_frame = reactive(0)

    def render(self) -> Group:
        if not self.conversation.messages:
            return Group(
                Panel(
                    Text.from_markup(
                        "[bold #eef4ff]Ready when you are.[/]\n"
                        "[#718096]Say the wake word to begin a conversation.[/]"
                    ),
                    border_style="#2d374b",
                    padding=(1, 2),
                )
            )

        return Group(
            *(self._render_message(message) for message in self.conversation.messages)
        )

    def advance_spinner(self) -> None:
        if any(
            isinstance(message, AssistantMessage)
            and bool(message.draft)
            and not message.interrupted
            for message in self.conversation.messages
        ):
            self.spinner_frame = (self.spinner_frame + 1) % len(_SPINNER_FRAMES)

    def watch_conversation(self) -> None:
        self.refresh(layout=True)

    def _render_message(
        self, message: UserMessage | AssistantMessage
    ) -> RenderableType:
        if isinstance(message, UserMessage):
            body = Text(message.text or "Listening\u2026", style="#eef4ff")
            if not message.committed:
                body.append("  \u25cf", style="blink bold #f72585")
            return Panel(
                body,
                title="[bold #4cc9f0]YOU[/]",
                title_align="left",
                border_style="#24506a",
                padding=(0, 2),
            )

        body = Text()
        for phrase in message.phrases:
            if body:
                body.append(" ")
            style = (
                _PHRASE_STYLES[PhraseState.DELIVERED]
                if message.interrupted
                else _PHRASE_STYLES[phrase.state]
            )
            body.append(phrase.text, style=style)
        if message.draft:
            if body:
                body.append(" ")
            if message.interrupted:
                body.append(
                    message.draft,
                    style=_PHRASE_STYLES[PhraseState.DELIVERED],
                )
            else:
                body.append(message.draft, style="italic #c084fc")
                body.append(
                    f"  {_SPINNER_FRAMES[self.spinner_frame]}",
                    style="bold #c084fc",
                )
        if message.interrupted:
            if body:
                body.append("\n")
            body.append(
                "\u2500\u2500  REPLY INTERRUPTED  \u2500\u2500", style="bold #ff6b7a"
            )
        return Panel(
            body if body else Text("Thinking\u2026", style="italic #b69cff"),
            title="[bold #42d3c7]HENRY[/]",
            title_align="left",
            border_style="#285650" if not message.interrupted else "#6b2b3a",
            padding=(0, 2),
        )


class ConversationView(Static):
    conversation = reactive(ConversationState())

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="conversation-scroll"):
            yield ConversationTranscript(id="conversation-transcript")
        yield Button("\u2193  Jump to latest", id="jump-latest", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#conversation-scroll", VerticalScroll).anchor()
        self.set_interval(0.08, self._advance_spinner)
        self.set_interval(0.2, self._update_jump_button)

    def watch_conversation(self, conversation: ConversationState) -> None:
        if not self.is_mounted:
            return
        self.query_one(ConversationTranscript).conversation = conversation

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "jump-latest":
            self.query_one("#conversation-scroll", VerticalScroll).scroll_end(
                animate=True
            )

    def _advance_spinner(self) -> None:
        self.query_one(ConversationTranscript).advance_spinner()

    def _update_jump_button(self) -> None:
        scroll = self.query_one("#conversation-scroll", VerticalScroll)
        self.query_one(
            "#jump-latest", Button
        ).display = not scroll.is_vertical_scroll_end


class LogsView(RichLog):
    pass


class ProgressDisplay(Static):
    snapshot = reactive(ProgressSnapshot())
    message = reactive("Preparing local models and services\u2026")

    def watch_snapshot(self) -> None:
        self.refresh(layout=True)

    def watch_message(self) -> None:
        self.refresh(layout=True)

    def render(self) -> Group:
        items: list[RenderableType] = [
            Text(self.message, style="bold #eef4ff"),
            Text("This may take a moment on first launch.\n", style="#718096"),
        ]
        if not self.snapshot.items:
            items.append(
                Text("  Waiting for model initialization\u2026", style="#b69cff")
            )
        else:
            active = [
                item
                for item in self.snapshot.items
                if item.status is ProgressStatus.ACTIVE
            ]
            completed = [
                item
                for item in self.snapshot.items
                if item.status is ProgressStatus.COMPLETED
            ]
            completed_limit = max(0, 8 - len(active))
            recent_completed = completed[-completed_limit:] if completed_limit else []
            visible = (*recent_completed, *active[-8:])
            items.extend(self._render_item(item) for item in visible)
        return Group(*items)

    @staticmethod
    def _render_item(item: ProgressItem) -> RenderableType:
        complete = item.status is ProgressStatus.COMPLETED
        percent = item.percentage
        heading = Text()
        heading.append(
            "\u2713 " if complete else "\u2193 ",
            style="#67e8a5" if complete else "#4cc9f0",
        )
        heading.append(_short(item.description, 54), style="#d8e0ee")
        if percent is not None:
            heading.append(f"  {percent:5.1f}%", style="bold #42d3c7")
        progress = ProgressBar(
            total=100,
            completed=100 if complete else percent or 0,
            width=44,
            style="#242c3d",
            complete_style="#42d3c7" if not complete else "#67e8a5",
            finished_style="#67e8a5",
        )
        return Group(heading, progress, Text())
