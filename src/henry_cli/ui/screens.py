from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, LoadingIndicator, OptionList, Static
from textual.widgets.option_list import Option

from henry_resources import Profile, ProfileEntry

from ..progress import ProgressSnapshot
from .widgets import ProgressDisplay


class ProfilePicker(ModalScreen[Profile | None]):
    _DEFAULT_PROFILE_ID = "default"
    _MISSING_PROFILE_ERROR = "Profile does not exist"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "cancel", "Quit"),
        Binding("escape", "cancel", show=False),
    ]

    def __init__(self, profiles: tuple[ProfileEntry, ...]) -> None:
        super().__init__()
        if not profiles:
            profiles = (
                ProfileEntry(
                    id=self._DEFAULT_PROFILE_ID,
                    name="Default",
                    error=self._MISSING_PROFILE_ERROR,
                ),
            )
        self._profiles = tuple(
            sorted(
                profiles,
                key=lambda entry: (
                    entry.id != self._DEFAULT_PROFILE_ID,
                    entry.id.casefold(),
                ),
            )
        )
        self._valid = {
            entry.id: entry.profile
            for entry in self._profiles
            if entry.profile is not None
        }

    def compose(self) -> ComposeResult:
        with Vertical(id="profile-dialog"):
            yield Static("HENRY  /  START", id="profile-kicker")
            yield Label("Choose your profile", id="profile-title")
            yield Static(
                "A valid profile is required before audio and models are initialized.",
                id="profile-subtitle",
            )
            options = [self._option(entry) for entry in self._profiles]
            if options:
                yield OptionList(*options, id="profile-options")
            else:
                yield Static(
                    "No profiles found. Add one below profiles/<profile-id>/.",
                    id="profiles-empty",
                )
            yield Static(
                "\u2191\u2193 navigate   ENTER select   Q quit", id="profile-help"
            )

    def on_option_list_option_selected(
        self,
        event: OptionList.OptionSelected,
    ) -> None:
        option_id = event.option_id
        if option_id is None:
            return
        profile = self._valid.get(option_id)
        if profile is not None:
            self.dismiss(profile)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @staticmethod
    def _option(entry: ProfileEntry) -> Option:
        prompt = Text()
        prompt.append("\u25c6  ", style="#42d3c7" if entry.is_valid else "#ff6b7a")
        prompt.append(entry.name, style="bold #eef4ff" if entry.is_valid else "#6f7785")
        prompt.append(f"\n   {entry.id}", style="#718096")
        if entry.error:
            error = " ".join(entry.error.splitlines())
            prompt.append(f"\n   INVALID  \u00b7  {error[:96]}", style="#9a4e5d")
        return Option(prompt, id=entry.id, disabled=not entry.is_valid)


class StartupScreen(ModalScreen[bool | None]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit_startup", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._error: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="startup-dialog"):
            yield Static("HENRY  /  INITIALIZING", id="startup-kicker")
            yield Label("Preparing your assistant", id="startup-title")
            yield LoadingIndicator(id="startup-spinner")
            yield ProgressDisplay(id="progress-display")
            yield Static("", id="startup-error")
            with Horizontal(id="startup-actions"):
                yield Button("Retry", id="retry-startup", variant="primary")
                yield Button("Quit", id="quit-startup", variant="error")

    def update_progress(self, snapshot: ProgressSnapshot) -> None:
        if self.is_mounted:
            self.query_one(ProgressDisplay).snapshot = snapshot

    def show_error(self, error: BaseException) -> None:
        self._error = str(error)
        if self.is_mounted:
            self.query_one("#startup-spinner", LoadingIndicator).display = False
            error_widget = self.query_one("#startup-error", Static)
            error_widget.update(f"STARTUP FAILED\n{self._error}")
            error_widget.display = True
            self.query_one("#startup-actions", Horizontal).display = True

    def reset(self) -> None:
        self._error = None
        if self.is_mounted:
            self.query_one("#startup-spinner", LoadingIndicator).display = True
            self.query_one("#startup-error", Static).display = False
            self.query_one("#startup-actions", Horizontal).display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "retry-startup":
            self.dismiss(True)
        elif event.button.id == "quit-startup":
            self.dismiss(False)

    def action_quit_startup(self) -> None:
        self.dismiss(False)
