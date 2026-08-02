import asyncio
from dataclasses import replace
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    ContentSwitcher,
    LoadingIndicator,
    OptionList,
    Static,
)

from henry_cli.events import UiEventBridge
from henry_cli.logs import LogBuffer
from henry_cli.progress import ProgressStore
from henry_cli.ui import TerminalApp
from henry_cli.ui.screens import ProfilePicker, StartupScreen
from henry_cli.ui.state import (
    AssistantMessage,
    AssistantPhrase,
    ConversationState,
    PhraseState,
    State,
    UserMessage,
)
from henry_cli.ui.widgets import (
    ConversationTranscript,
    ConversationView,
    HeaderBar,
    InfoPanel,
    ProgressDisplay,
    _meter,
    _short,
)
from henry_conversation import ConversationReady
from henry_resources import ProfileEntry, Settings
from henry_speech.events import UserTurnCommitted

from .test_state import profile


def rendered(renderable) -> str:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=100)
    console.print(renderable)
    return stream.getvalue()


def test_dashboard_renderables_expose_runtime_and_conversation_states() -> None:
    state = State().with_runtime(profile(), Settings())
    header = HeaderBar()
    header.state = state
    assert "HENRY" in rendered(header.render())

    panel = InfoPanel()
    panel.state = replace(
        state,
        telemetry=replace(
            state.telemetry,
            vad_score=1.2,
            vad_detected=True,
            wakeword_score=-1,
            timings=(("reply_started", 99.9),),
        ),
    )
    info = rendered(panel.render())
    assert "SIGNALS" in info
    assert "REPLY STARTED" in info
    assert "test/fast" in info
    assert "█" in rendered(_meter(0.5, False))
    assert _short("short") == "short"
    assert _short("very long value", 8).endswith("…")

    transcript = ConversationTranscript()
    assert "Ready when you are" in rendered(transcript.render())
    interrupted = AssistantMessage(
        1,
        phrases=(
            AssistantPhrase(1, "Queued.", PhraseState.QUEUED),
            AssistantPhrase(2, "Speaking.", PhraseState.SPEAKING),
            AssistantPhrase(3, "Delivered.", PhraseState.DELIVERED),
        ),
        draft="Building",
        interrupted=True,
    )
    transcript.conversation = ConversationState(
        (UserMessage(1, "Question", committed=False), interrupted)
    )
    output = rendered(transcript.render())
    assert "Question" in output
    assert "Queued. Speaking. Delivered. Building" in output
    assert "REPLY INTERRUPTED" in output
    assert not any(
        frame in output
        for frame in "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
    )
    interrupted_panel = transcript._render_message(interrupted)
    assert isinstance(interrupted_panel, Panel)
    assert isinstance(interrupted_panel.renderable, Text)
    assert {str(span.style) for span in interrupted_panel.renderable.spans[:-1]} == {
        "#a7b0c0"
    }
    transcript.advance_spinner()
    assert transcript.spinner_frame == 0

    transcript.conversation = ConversationState(
        (UserMessage(2, "Done", committed=True), AssistantMessage(2))
    )
    assert "Thinking" in rendered(transcript.render())
    transcript.advance_spinner()
    assert transcript.spinner_frame == 0
    transcript.conversation = ConversationState((AssistantMessage(3, draft="Live"),))
    transcript.advance_spinner()
    assert transcript.spinner_frame == 1


def test_progress_renderable_handles_waiting_active_and_completed_items() -> None:
    display = ProgressDisplay()
    assert "Waiting for model initialization" in rendered(display.render())
    store = ProgressStore()
    active = store.begin("weights.safetensors", 10, 100, "B")
    store.begin("metadata", 0, None, "files")
    display.snapshot = store.snapshot
    assert "10.0%" in rendered(display.render())
    store.complete(active, 100, 100)
    display.snapshot = store.snapshot
    assert "✓" in rendered(display.render())


def test_terminal_app_profile_startup_navigation_and_live_updates() -> None:
    async def scenario() -> None:
        selected_profile = profile()
        profiles = (
            ProfileEntry("valid", "Valid", selected_profile),
            ProfileEntry("invalid", "Invalid", error="broken profile"),
        )
        bridge = UiEventBridge()
        logs = LogBuffer()
        progress = ProgressStore()
        app = TerminalApp(profiles, bridge, logs, progress)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            picker = app.screen
            assert isinstance(picker, ProfilePicker)
            options = picker.query_one(OptionList)
            assert options.option_count == 2
            assert options.get_option("invalid").disabled
            await pilot.press("enter")
            assert await asyncio.wait_for(app.select_profile(), 1) is selected_profile

            app.configure_runtime(selected_profile, Settings())
            await app.show_startup()
            assert isinstance(app.screen, StartupScreen)
            item = progress.begin("model.bin", 40, 100, "B")
            app._refresh_progress()
            assert app.screen.query_one(ProgressDisplay).snapshot.is_active
            progress.complete(item, 100, 100)

            retry_task = asyncio.create_task(
                app.wait_startup_retry(RuntimeError("download failed"))
            )
            await pilot.pause()
            startup = app.screen
            assert isinstance(startup, StartupScreen)
            assert not startup.query_one(LoadingIndicator).display
            assert startup.query_one("#startup-error", Static).display
            assert startup.query_one("#startup-actions", Horizontal).display
            startup.query_one("#retry-startup", Button).press()
            assert await asyncio.wait_for(retry_task, 1)

            await app.show_startup()
            await app.finish_startup()
            await pilot.pause()

            app.action_show_logs()
            assert app.query_one(ContentSwitcher).current == "logs"
            logs.write("hello from worker")
            app._flush_logs()
            app.action_show_conversation()
            assert app.query_one(ContentSwitcher).current == "conversation"

            app.action_toggle_info()
            await pilot.pause(0.2)
            assert not app.query_one(InfoPanel).display
            app.action_toggle_info()
            await pilot.pause(0.25)
            assert app.query_one(InfoPanel).display

            bridge._queue.put_nowait(ConversationReady())
            bridge._queue.put_nowait(UserTurnCommitted(1, "Visible question"))
            await pilot.pause()
            assert app.state.conversation_ready
            await bridge._queue.join()

            view = app.query_one(ConversationView)
            view._advance_spinner()
            view._update_jump_button()
            view.query_one("#jump-latest", Button).press()
            await pilot.pause()

            app.action_request_quit()
            await asyncio.wait_for(app.wait_quit_requested(), 1)

    asyncio.run(scenario())


def test_conversation_follows_latest_until_user_scrolls_history() -> None:
    class ConversationApp(App[None]):
        def compose(self) -> ComposeResult:
            yield ConversationView()

    async def scenario() -> None:
        app = ConversationApp()
        async with app.run_test(size=(70, 12)) as pilot:
            view = app.query_one(ConversationView)
            scroll = view.query_one("#conversation-scroll", VerticalScroll)
            assert scroll.is_anchored

            messages = tuple(
                UserMessage(index, f"Message {index}", committed=True)
                for index in range(12)
            )
            view.conversation = ConversationState(messages)
            await pilot.pause()
            assert scroll.is_vertical_scroll_end

            view.conversation = ConversationState(
                (*messages, AssistantMessage(13, draft="A growing reply " * 20))
            )
            await pilot.pause()
            assert scroll.is_vertical_scroll_end

            scroll.scroll_up(animate=False, immediate=True)
            history_position = scroll.scroll_y
            view.conversation = ConversationState(
                (
                    *messages,
                    AssistantMessage(13, draft="A growing reply " * 40),
                )
            )
            await pilot.pause()
            assert scroll.scroll_y == history_position
            assert not scroll.is_vertical_scroll_end
            assert view.query_one("#jump-latest", Button).display

            view.query_one("#jump-latest", Button).press()
            await pilot.pause(1.1)
            assert scroll.is_vertical_scroll_end

    asyncio.run(scenario())


def test_startup_screen_operations_are_marshaled_to_textual_task() -> None:
    async def scenario() -> None:
        app = TerminalApp((), UiEventBridge(), LogBuffer(), ProgressStore())
        app_task = asyncio.create_task(
            app.run_async(headless=True, size=(90, 30)),
            name="test-ui",
        )
        try:
            await asyncio.wait_for(app.wait_mounted(), 1)
            await asyncio.wait_for(app.show_startup(), 1)
            assert isinstance(app.screen, StartupScreen)
            await asyncio.wait_for(app.finish_startup(), 1)
            assert isinstance(app.screen, ProfilePicker)
        finally:
            app.exit()
            await asyncio.wait_for(app_task, 1)

    asyncio.run(scenario())


def test_profile_and_startup_modal_cancel_paths() -> None:
    async def scenario() -> None:
        app = TerminalApp((), UiEventBridge(), LogBuffer(), ProgressStore())
        async with app.run_test(size=(90, 30)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ProfilePicker)
            options = app.screen.query_one(OptionList)
            assert options.option_count == 1
            fallback = options.get_option_at_index(0)
            assert fallback.id == "default"
            assert fallback.disabled
            app.action_request_quit()
            assert await asyncio.wait_for(app.select_profile(), 1) is None

        profile_app = TerminalApp(
            (ProfileEntry("valid", "Valid", profile()),),
            UiEventBridge(),
            LogBuffer(),
            ProgressStore(),
        )
        async with profile_app.run_test(size=(90, 30)) as pilot:
            await pilot.pause()
            profile_app.screen.action_cancel()
            assert await asyncio.wait_for(profile_app.select_profile(), 1) is None

        startup_app = TerminalApp(
            (ProfileEntry("valid", "Valid", profile()),),
            UiEventBridge(),
            LogBuffer(),
            ProgressStore(),
        )
        async with startup_app.run_test(size=(90, 30)) as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await startup_app.select_profile()
            await startup_app.show_startup()
            startup = startup_app.screen
            assert isinstance(startup, StartupScreen)
            startup.show_error(RuntimeError("failed"))
            startup.reset()
            assert startup.query_one(LoadingIndicator).display
            startup.action_quit_startup()

    asyncio.run(scenario())


def test_profile_picker_places_default_first() -> None:
    async def scenario() -> None:
        entries = (
            ProfileEntry("zeta", "Zeta", profile()),
            ProfileEntry("default", "Henry", profile()),
            ProfileEntry("alpha", "Alpha", profile()),
        )
        app = TerminalApp(entries, UiEventBridge(), LogBuffer(), ProgressStore())
        async with app.run_test(size=(90, 30)) as pilot:
            await pilot.pause()
            options = app.screen.query_one(OptionList)
            assert [
                options.get_option_at_index(index).id
                for index in range(options.option_count)
            ] == ["default", "alpha", "zeta"]

    asyncio.run(scenario())


def test_conversation_view_ignores_updates_before_mount() -> None:
    view = ConversationView()
    conversation = ConversationState((UserMessage(1, "Hello"),))
    view.watch_conversation(conversation)
