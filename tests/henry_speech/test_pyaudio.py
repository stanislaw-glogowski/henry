import pytest

from henry_speech.audio import AudioDevice, AudioDevices
from henry_speech.audio.adapters.pyaudio import PyAudioDriver
from henry_speech.audio.adapters.pyaudio import driver as driver_module


def test_pyaudio_driver_exposes_default_devices_and_streams(monkeypatch) -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.terminated = False

        def get_default_input_device_info(self) -> dict:
            return {"index": 1, "name": "MacBook Microphone"}

        def get_default_output_device_info(self) -> dict:
            return {"index": 2, "name": "Studio Display"}

        def terminate(self) -> None:
            self.terminated = True

    session = FakeSession()
    monkeypatch.setattr(driver_module.pyaudio, "PyAudio", lambda: session)
    driver = PyAudioDriver()

    with pytest.raises(RuntimeError, match="not open"):
        _ = driver.devices

    with driver:
        assert driver.devices == AudioDevices(
            input=AudioDevice("MacBook Microphone", "1"),
            output=AudioDevice("Studio Display", "2"),
        )
        assert driver.input is driver.input
        assert driver.output is driver.output

    assert session.terminated
    with pytest.raises(RuntimeError, match="not open"):
        _ = driver.input
