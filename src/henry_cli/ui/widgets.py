from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import RichLog, Static

from ..events import TelemetrySnapshot
from ..progress import ProgressItem, ProgressSnapshot, ProgressStatus
from .state import (
    AssistantMessage,
    ConversationState,
    PhraseState,
    RuntimeInfo,
    RuntimeMode,
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
    mode = reactive(RuntimeMode.STARTING)
    profile_name = reactive("No profile")

    def render(self) -> Text:
        status = Text(" \u25cf ", style=_MODE_STYLES[self.mode])
        status.append(self.mode.value, style=_MODE_STYLES[self.mode])
        status.append("  \u00b7  ", style="#465064")
        status.append(self.profile_name, style="#b6c2d6")
        return status


class SignalsPanel(Static):
    telemetry = reactive(TelemetrySnapshot())

    def render(self) -> Group:
        telemetry = self.telemetry
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
        return Group(Text("SIGNALS", style="bold #42d3c7"), status)


class RuntimePanel(Static):
    info = reactive(RuntimeInfo())

    def render(self) -> Group:
        info = self.info
        runtime = Table.grid(expand=True, padding=(0, 1))
        runtime.add_column(width=10, style="#64748b")
        runtime.add_column(ratio=1, overflow="fold")
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
        return Group(Text("RUNTIME", style="bold #b69cff"), runtime)


class LatencyPanel(Static):
    timings = reactive(TelemetrySnapshot().timings)

    def render(self) -> Group:
        timing = Table.grid(expand=True, padding=(0, 1))
        timing.add_column(ratio=1, style="#718096")
        timing.add_column(width=10, justify="right", style="#b69cff")
        if self.timings:
            for stage, elapsed_ms in self.timings:
                timing.add_row(stage.replace("_", " ").upper(), f"{elapsed_ms:,.0f} ms")
        else:
            timing.add_row("NO INTERACTION YET", "\u2014")
        return Group(Text("LATENCY", style="bold #ffd166"), timing)


class InfoPanel(Static):
    info = reactive(RuntimeInfo(), repaint=False)
    telemetry = reactive(TelemetrySnapshot(), repaint=False)

    def compose(self) -> ComposeResult:
        yield SignalsPanel(id="info-signals")
        yield RuntimePanel(id="info-runtime")
        yield LatencyPanel(id="info-latency")

    def on_mount(self) -> None:
        self.watch_info(self.info)
        self.watch_telemetry(self.telemetry)

    def watch_info(self, info: RuntimeInfo) -> None:
        if self.is_mounted:
            self.query_one(RuntimePanel).info = info

    def watch_telemetry(self, telemetry: TelemetrySnapshot) -> None:
        if not self.is_mounted:
            return
        self.query_one(SignalsPanel).telemetry = telemetry
        self.query_one(LatencyPanel).timings = telemetry.timings


type ConversationMessage = UserMessage | AssistantMessage


class ConversationMessageView(Static):
    def __init__(
        self,
        message: ConversationMessage,
        spinner_frame: int = 0,
        assistant_name: str = "Assistant",
    ) -> None:
        super().__init__(classes="conversation-message")
        self.message = message
        self._spinner_frame = spinner_frame
        self._assistant_name = assistant_name

    def render(self) -> Panel:
        message = self.message
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
                    f"  {_SPINNER_FRAMES[self._spinner_frame]}",
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
            title=Text(self._assistant_name, style="bold #42d3c7"),
            title_align="left",
            border_style="#285650" if not message.interrupted else "#6b2b3a",
            padding=(0, 2),
        )

    def update_message(
        self,
        message: ConversationMessage,
        spinner_frame: int,
        assistant_name: str,
        *,
        layout: bool = True,
    ) -> None:
        if (
            self.message == message
            and self._spinner_frame == spinner_frame
            and self._assistant_name == assistant_name
        ):
            return
        self.message = message
        self._spinner_frame = spinner_frame
        self._assistant_name = assistant_name
        self.refresh(layout=layout)


class ConversationEmptyView(Static):
    wakeword_label = reactive("wake word")

    def render(self) -> Panel:
        body = Text("Waiting for ", style="#718096")
        body.append(f"“{self.wakeword_label}”", style="bold #42d3c7")
        body.append("…", style="#718096")
        return Panel(
            body,
            title=Text("WAKE WORD", style="bold #b69cff"),
            title_align="left",
            border_style="#2d374b",
            padding=(1, 2),
        )


class ConversationTranscript(Vertical):
    conversation = reactive(ConversationState(), repaint=False)
    assistant_name = reactive("Assistant", repaint=False)
    wakeword_label = reactive("wake word", repaint=False)
    waiting_for_wakeword = reactive(True, repaint=False)
    spinner_frame = 0

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._message_views: dict[tuple[str, int], ConversationMessageView] = {}

    def compose(self) -> ComposeResult:
        yield ConversationEmptyView(id="conversation-empty")

    async def on_mount(self) -> None:
        self.watch_wakeword_label(self.wakeword_label)
        await self.watch_conversation(self.conversation)

    async def watch_conversation(self, conversation: ConversationState) -> None:
        if not self.is_mounted:
            return
        self._update_empty_view()
        desired_keys = {self._message_key(message) for message in conversation.messages}
        for key, widget in tuple(self._message_views.items()):
            if key not in desired_keys:
                await widget.remove()
                del self._message_views[key]

        for message in conversation.messages:
            key = self._message_key(message)
            spinner_frame = self._message_spinner_frame(message)
            if widget := self._message_views.get(key):
                widget.update_message(message, spinner_frame, self.assistant_name)
            else:
                widget = ConversationMessageView(
                    message,
                    spinner_frame,
                    self.assistant_name,
                )
                self._message_views[key] = widget
                await self.mount(widget)

    def watch_assistant_name(self, assistant_name: str) -> None:
        for widget in self._message_views.values():
            widget.update_message(
                widget.message,
                widget._spinner_frame,
                assistant_name,
                layout=False,
            )

    def watch_wakeword_label(self, wakeword_label: str) -> None:
        if self.is_mounted:
            self.query_one(ConversationEmptyView).wakeword_label = wakeword_label

    def watch_waiting_for_wakeword(self) -> None:
        if self.is_mounted:
            self._update_empty_view()

    def _update_empty_view(self) -> None:
        self.query_one(ConversationEmptyView).display = (
            not self.conversation.messages and self.waiting_for_wakeword
        )

    @staticmethod
    def _message_key(message: ConversationMessage) -> tuple[str, int]:
        if isinstance(message, UserMessage):
            return "user", message.turn_id
        return "assistant", message.reply_id

    def _message_spinner_frame(self, message: ConversationMessage) -> int:
        if (
            isinstance(message, AssistantMessage)
            and message.draft
            and not message.interrupted
        ):
            return self.spinner_frame
        return 0

    def advance_spinner(self) -> None:
        message = next(
            (
                item
                for item in reversed(self.conversation.messages)
                if isinstance(item, AssistantMessage)
                and item.draft
                and not item.interrupted
            ),
            None,
        )
        if message is None:
            return
        self.spinner_frame = (self.spinner_frame + 1) % len(_SPINNER_FRAMES)
        key = self._message_key(message)
        if widget := self._message_views.get(key):
            widget.update_message(
                message,
                self.spinner_frame,
                self.assistant_name,
                layout=False,
            )


class ConversationView(Static):
    conversation = reactive(ConversationState(), repaint=False)
    assistant_name = reactive("Assistant", repaint=False)
    wakeword_label = reactive("wake word", repaint=False)
    waiting_for_wakeword = reactive(True, repaint=False)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="conversation-scroll"):
            yield ConversationTranscript(id="conversation-transcript")

    def on_mount(self) -> None:
        self.query_one("#conversation-scroll", VerticalScroll).anchor()
        self.set_interval(0.08, self._advance_spinner)

    def watch_conversation(self, conversation: ConversationState) -> None:
        if not self.is_mounted:
            return
        self.query_one(ConversationTranscript).conversation = conversation

    def watch_assistant_name(self, assistant_name: str) -> None:
        if self.is_mounted:
            self.query_one(ConversationTranscript).assistant_name = assistant_name

    def watch_wakeword_label(self, wakeword_label: str) -> None:
        if self.is_mounted:
            self.query_one(ConversationTranscript).wakeword_label = wakeword_label

    def watch_waiting_for_wakeword(self, waiting: bool) -> None:
        if self.is_mounted:
            self.query_one(ConversationTranscript).waiting_for_wakeword = waiting

    def _advance_spinner(self) -> None:
        self.query_one(ConversationTranscript).advance_spinner()


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
