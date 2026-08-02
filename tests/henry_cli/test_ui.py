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
    ConversationMessageView,
    ConversationTranscript,
    ConversationView,
    HeaderBar,
    InfoPanel,
    LatencyPanel,
    ProgressDisplay,
    RuntimePanel,
    SignalsPanel,
    _meter,
    _short,
)
from henry_conversation import ConversationReady
from henry_resources import ProfileEntry, Settings
from henry_speech.events import (
    UserTurnCommitted,
    VoiceSessionMode,
    VoiceSessionModeChanged,
)

from .test_state import profile


def rendered(renderable) -> str:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=100)
    console.print(renderable)
    return stream.getvalue()


def test_dashboard_renderables_expose_runtime_and_conversation_states() -> None:
    state = State().with_runtime(profile(), Settings())
    header = HeaderBar()
    header.mode = state.mode
    header.profile_name = state.info.profile_name
    assert "HENRY" in rendered(header.render())

    telemetry = replace(
        state,
        telemetry=replace(
            state.telemetry,
            vad_score=1.2,
            vad_detected=True,
            wakeword_score=-1,
            timings=(("reply_started", 99.9),),
        ),
    ).telemetry
    signals = SignalsPanel()
    signals.telemetry = telemetry
    runtime = RuntimePanel()
    runtime.info = state.info
    latency = LatencyPanel()
    latency.timings = telemetry.timings
    info = "".join(rendered(panel.render()) for panel in (signals, runtime, latency))
    assert "SIGNALS" in info
    assert "PROFILE" not in rendered(runtime.render())
    assert "REPLY STARTED" in info
    assert "test/fast" in info
    assert "█" in rendered(_meter(0.5, False))
    assert _short("short") == "short"
    assert _short("very long value", 8).endswith("…")

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
    interrupted_view = ConversationMessageView(
        interrupted,
        assistant_name="Test Henry",
    )
    output = rendered(interrupted_view.render())
    assert "Queued. Speaking. Delivered. Building" in output
    assert "REPLY INTERRUPTED" in output
    assert not any(
        frame in output
        for frame in "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
    )
    interrupted_panel = interrupted_view.render()
    assert isinstance(interrupted_panel, Panel)
    assert isinstance(interrupted_panel.title, Text)
    assert interrupted_panel.title.plain == "Test Henry"
    assert isinstance(interrupted_panel.renderable, Text)
    assert {str(span.style) for span in interrupted_panel.renderable.spans[:-1]} == {
        "#a7b0c0"
    }
    assert "Thinking" in rendered(ConversationMessageView(AssistantMessage(2)).render())
    live = ConversationMessageView(AssistantMessage(3, draft="Live"), spinner_frame=1)
    assert "Live" in rendered(live.render())


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

            view = app.query_one(ConversationView)
            empty = view.query_one("#conversation-empty", Static)
            assert "Waiting for “Wake”…" in rendered(empty.render())
            bridge._queue.put_nowait(VoiceSessionModeChanged(VoiceSessionMode.ACTIVE))
            await bridge._queue.join()
            await pilot.pause()
            assert not empty.display

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

            view._advance_spinner()
            await pilot.pause()

            app.action_request_quit()
            await asyncio.wait_for(app.wait_quit_requested(), 1)

    asyncio.run(scenario())


def test_conversation_follows_latest_until_user_scrolls_history() -> None:
    class ConversationApp(App[None]):
        CSS = """
        ConversationView, #conversation-scroll { height: 100%; }
        #conversation-transcript, .conversation-message { height: auto; }
        """

        def compose(self) -> ComposeResult:
            yield ConversationView()

    async def scenario() -> None:
        app = ConversationApp()
        async with app.run_test(size=(70, 12)) as pilot:
            view = app.query_one(ConversationView)
            scroll = view.query_one("#conversation-scroll", VerticalScroll)
            transcript = view.query_one(ConversationTranscript)
            assert scroll.is_anchored
            assert transcript.query_one("#conversation-empty", Static).display
            view.assistant_name = "Test Henry"
            view.wakeword_label = "Hey Henry"
            await pilot.pause()
            empty = transcript.query_one("#conversation-empty", Static)
            assert "Waiting for “Hey Henry”…" in rendered(empty.render())
            view.waiting_for_wakeword = False
            await pilot.pause()
            assert not empty.display
            view.waiting_for_wakeword = True

            messages = tuple(
                UserMessage(index, f"Message {index}", committed=True)
                for index in range(12)
            )
            view.conversation = ConversationState(messages)
            await pilot.pause()
            assert scroll.is_vertical_scroll_end
            original_widgets = tuple(transcript.query(ConversationMessageView))
            assert len(original_widgets) == len(messages)

            view.conversation = ConversationState(
                (*messages, AssistantMessage(13, draft="A growing reply " * 20))
            )
            await pilot.pause()
            assert scroll.is_vertical_scroll_end
            growing_widgets = tuple(transcript.query(ConversationMessageView))
            assert growing_widgets[:-1] == original_widgets
            active_reply = growing_widgets[-1]
            active_panel = active_reply.render()
            assert isinstance(active_panel.title, Text)
            assert active_panel.title.plain == "Test Henry"

            view.conversation = ConversationState(
                (*messages, AssistantMessage(13, draft="A longer growing reply"))
            )
            await pilot.pause()
            updated_widgets = tuple(transcript.query(ConversationMessageView))
            assert updated_widgets == growing_widgets
            assert active_reply.message == AssistantMessage(
                13, draft="A longer growing reply"
            )
            spinner_frame = active_reply._spinner_frame
            transcript.advance_spinner()
            assert active_reply._spinner_frame == (spinner_frame + 1) % 10

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
            scroll.scroll_end(animate=False)
            await pilot.pause()
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
