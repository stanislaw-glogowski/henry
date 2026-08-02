from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import ContentSwitcher, Footer

from henry_resources import Profile, ProfileEntry, Settings
from henry_speech.events import VoiceSessionMode

from ..events import UiEventBridge
from ..logs import LogBuffer
from ..progress import ProgressStore
from .screens import ProfilePicker, StartupScreen
from .state import State
from .widgets import ConversationView, HeaderBar, InfoPanel, LogsView


class TerminalApp(App[None]):
    CSS_PATH = "styles.tcss"
    ENABLE_COMMAND_PALETTE = False
    TITLE = "Henry"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("c", "show_conversation", "Conversation"),
        Binding("l", "show_logs", "Logs"),
        Binding("i", "toggle_info", "Info"),
        Binding("q", "request_quit", "Quit"),
    ]

    state = reactive(State(), repaint=False)

    def __init__(
        self,
        profiles: tuple[ProfileEntry, ...],
        events: UiEventBridge,
        logs: LogBuffer,
        progress: ProgressStore,
    ) -> None:
        super().__init__()
        self._profiles = profiles
        self._events = events
        self._logs = logs
        self._progress = progress
        self._profile_queue: asyncio.Queue[Profile | None] = asyncio.Queue(maxsize=1)
        self._startup_queue: asyncio.Queue[bool | None] = asyncio.Queue(maxsize=1)
        self._startup_screen: StartupScreen | None = None
        self._mounted = asyncio.Event()
        self._quit_requested = asyncio.Event()

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header-bar")
        with Horizontal(id="workspace"):
            yield InfoPanel(id="info-panel")
            with ContentSwitcher(initial="conversation", id="content"):
                yield ConversationView(id="conversation")
                yield LogsView(id="logs", highlight=True, markup=False, wrap=True)
        yield Footer(show_command_palette=False)

    async def wait_mounted(self) -> None:
        await self._mounted.wait()

    async def select_profile(self) -> Profile | None:
        return await self._profile_queue.get()

    async def wait_quit_requested(self) -> None:
        await self._quit_requested.wait()

    def configure_runtime(self, profile: Profile, settings: Settings) -> None:
        self.state = self.state.with_runtime(profile, settings)

    async def show_startup(self) -> None:
        await self._run_in_app(self._show_startup)

    async def _show_startup(self) -> None:
        screen = StartupScreen()
        self._startup_screen = screen
        await self.push_screen(screen, self._startup_finished)

    async def wait_startup_retry(self, error: BaseException) -> bool:
        screen = self._startup_screen
        if screen is None:
            return False
        screen.show_error(error)
        result = await self._startup_queue.get()
        self._startup_screen = None
        return bool(result)

    async def finish_startup(self) -> None:
        await self._run_in_app(self._finish_startup)

    async def _finish_startup(self) -> None:
        screen = self._startup_screen
        if screen is not None and self.screen is screen:
            await self.pop_screen()
        self._startup_screen = None

    async def _run_in_app[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        future = asyncio.get_running_loop().create_future()

        async def run() -> None:
            try:
                result = await operation()
            except BaseException as error:
                if not future.done():
                    future.set_exception(error)
            else:
                if not future.done():
                    future.set_result(result)

        if not self.call_later(run):
            raise RuntimeError("Textual application is not running")
        return await future

    def action_show_conversation(self) -> None:
        self.query_one(ContentSwitcher).current = "conversation"

    def action_show_logs(self) -> None:
        self.query_one(ContentSwitcher).current = "logs"

    def action_toggle_info(self) -> None:
        panel = self.query_one("#info-panel", InfoPanel)
        if panel.display:
            panel.styles.animate(
                "opacity",
                0.0,
                duration=0.16,
                on_complete=lambda: setattr(panel, "display", False),
            )
        else:
            panel.display = True
            panel.styles.opacity = 0.0
            panel.styles.animate("opacity", 1.0, duration=0.2)

    def action_request_quit(self) -> None:
        if isinstance(self.screen, ProfilePicker):
            self.screen.action_cancel()
            return
        self._quit_requested.set()

    def watch_state(self, previous: State, state: State) -> None:
        if not self.is_mounted:
            return
        header = self.query_one(HeaderBar)
        if previous.mode is not state.mode:
            header.mode = state.mode
        if previous.info.profile_name != state.info.profile_name:
            header.profile_name = state.info.profile_name
        info = self.query_one(InfoPanel)
        conversation = self.query_one(ConversationView)
        if previous.info != state.info:
            info.info = state.info
        if previous.info.profile_name != state.info.profile_name:
            conversation.assistant_name = state.info.profile_name
        if previous.info.wakeword_label != state.info.wakeword_label:
            conversation.wakeword_label = state.info.wakeword_label
        if previous.telemetry != state.telemetry:
            info.telemetry = state.telemetry
        if previous.session_mode is not state.session_mode:
            conversation.waiting_for_wakeword = (
                state.session_mode is not VoiceSessionMode.ACTIVE
            )
        if previous.conversation != state.conversation:
            conversation.conversation = state.conversation

    def on_mount(self) -> None:
        self._mounted.set()
        self._consume_events()
        self.set_interval(0.08, self._refresh_telemetry)
        self.set_interval(0.08, self._flush_logs)
        self.set_interval(0.1, self._refresh_progress)
        self.push_screen(ProfilePicker(self._profiles), self._profile_selected)

    def _profile_selected(self, profile: Profile | None) -> None:
        self._profile_queue.put_nowait(profile)

    def _startup_finished(self, result: bool | None) -> None:
        self._startup_screen = None
        if result is not None:
            self._startup_queue.put_nowait(result)

    @work(group="events", exclusive=True)
    async def _consume_events(self) -> None:
        while True:
            event = await self._events.receive()
            try:
                self.state = self.state.reduce_event(event)
            finally:
                self._events.task_done()

    def _refresh_telemetry(self) -> None:
        self.state = self.state.with_telemetry(self._events.telemetry_snapshot)

    def _flush_logs(self) -> None:
        try:
            widget = self.query_one("#logs", LogsView)
        except NoMatches:
            return
        for line in self._logs.drain():
            widget.write(line)

    def _refresh_progress(self) -> None:
        screen = self._startup_screen
        if screen is not None and self.screen is screen:
            screen.update_progress(self._progress.snapshot)
