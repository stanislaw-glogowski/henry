from abc import ABC, abstractmethod

from henry_common.components import AbstractResource

from .domain import AudioDevices, AudioFrame, AudioPlaybackOutcome


class AudioInput(AbstractResource, ABC):
    @abstractmethod
    def read(self) -> AudioFrame:
        raise NotImplementedError


class AudioOutput(AbstractResource, ABC):
    @abstractmethod
    def write(self, frame: AudioFrame) -> AudioPlaybackOutcome:
        raise NotImplementedError

    def interrupt(self) -> None:
        """Stop queued and active playback when the adapter supports it."""

    def duck(self) -> None:
        """Temporarily lower playback while potential user speech is confirmed."""

    def restore(self) -> None:
        """Restore playback after a false barge-in detection."""


class AudioDriver[TInput = AudioInput, TOutput = AudioOutput](AbstractResource, ABC):
    @property
    @abstractmethod
    def input(self) -> TInput:
        raise NotImplementedError

    @property
    @abstractmethod
    def output(self) -> TOutput:
        raise NotImplementedError

    @property
    @abstractmethod
    def devices(self) -> AudioDevices:
        """Return devices selected for the currently open driver session."""
        raise NotImplementedError
