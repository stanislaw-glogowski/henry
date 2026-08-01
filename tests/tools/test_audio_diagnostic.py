from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from henry_speech.audio import AudioFormat, AudioPlaybackOutcome
from tools.audio_diagnostic.cli import (
    main,
    measure_full_duplex,
    record_and_play,
)


class FakePort:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def open(self) -> None:
        self.calls.append(f"{self.__class__.__name__}.open")

    def close(self) -> None:
        self.calls.append(f"{self.__class__.__name__}.close")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class FakeInput(FakePort):
    def __init__(self, calls: list[str]) -> None:
        super().__init__(calls)
        self._format = AudioFormat(sample_rate=4, channels=1)

    def read(self):
        self.calls.append("read")
        return self._format.build_frame(np.asarray([0.25, -0.5], dtype=np.float32))


class FakeOutput(FakePort):
    def __init__(
        self,
        calls: list[str],
        outcome: AudioPlaybackOutcome = AudioPlaybackOutcome.PLAYED,
    ) -> None:
        super().__init__(calls)
        self.outcome = outcome
        self.frame = None

    def write(self, frame):
        self.calls.append("write")
        self.frame = frame
        return self.outcome


class FakeDriver(FakePort):
    def __init__(
        self,
        calls: list[str],
        outcome: AudioPlaybackOutcome = AudioPlaybackOutcome.PLAYED,
    ) -> None:
        super().__init__(calls)
        self.input = FakeInput(calls)
        self.output = FakeOutput(calls, outcome)


def test_record_and_play_uses_port_lifecycle_and_exact_duration(capsys) -> None:
    calls: list[str] = []
    driver = FakeDriver(calls)

    recorded = record_and_play(driver, 0.75)  # type: ignore[arg-type]

    np.testing.assert_array_equal(recorded.samples, [0.25, -0.5, 0.25])
    assert driver.output.frame is recorded
    assert calls == [
        "FakeDriver.open",
        "FakeInput.open",
        "FakeOutput.open",
        "read",
        "read",
        "write",
        "FakeOutput.close",
        "FakeInput.close",
        "FakeDriver.close",
    ]
    output = capsys.readouterr().out
    assert "rms=" in output
    assert "Playback completed" in output


def test_record_and_play_validates_duration_and_interruption() -> None:
    with pytest.raises(ValueError, match="positive"):
        record_and_play(FakeDriver([]), 0)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="interrupted"):
        record_and_play(  # type: ignore[arg-type]
            FakeDriver([], AudioPlaybackOutcome.INTERRUPTED),
            0.25,
        )


def test_measure_full_duplex_captures_and_replays_residual(capsys) -> None:
    calls: list[str] = []
    driver = FakeDriver(calls)

    residual = measure_full_duplex(driver, 0.25)  # type: ignore[arg-type]

    np.testing.assert_array_equal(residual.samples, [0.25])
    assert calls.count("read") == 2
    assert calls.count("write") == 2
    output = capsys.readouterr().out
    assert "relative_to_playback=" in output
    assert "Full-duplex diagnostic completed" in output


def test_measure_full_duplex_validates_duration_and_interruption() -> None:
    with pytest.raises(ValueError, match="positive"):
        measure_full_duplex(FakeDriver([]), 0)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="Test-signal playback was interrupted"):
        measure_full_duplex(  # type: ignore[arg-type]
            FakeDriver([], AudioPlaybackOutcome.INTERRUPTED),
            0.25,
        )


def test_main_uses_configured_or_explicit_driver(monkeypatch) -> None:
    selected = []
    driver = object()
    store = SimpleNamespace(
        load_settings=lambda: SimpleNamespace(
            speech=SimpleNamespace(audio=SimpleNamespace(driver="avfaudio"))
        )
    )
    monkeypatch.setattr("tools.audio_diagnostic.cli.LocalStore", lambda: store)
    monkeypatch.setattr(
        "tools.audio_diagnostic.cli.get_audio_driver",
        lambda settings: selected.append(settings.driver) or driver,
    )
    monkeypatch.setattr(
        "tools.audio_diagnostic.cli.record_and_play",
        lambda actual_driver, seconds: selected.extend((actual_driver, seconds)),
    )
    monkeypatch.setattr(
        "tools.audio_diagnostic.cli.measure_full_duplex",
        lambda actual_driver, seconds: selected.extend(
            ("duplex", actual_driver, seconds)
        ),
    )

    main(["--seconds", "2"])
    main(["--seconds", "3", "--driver", "pyaudio"])
    main(["--seconds", "4", "--duplex"])

    assert selected == [
        "avfaudio",
        driver,
        2.0,
        "pyaudio",
        driver,
        3.0,
        "avfaudio",
        "duplex",
        driver,
        4.0,
    ]
