import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from henry_speech.audio import (
    AudioDevice,
    AudioDevices,
    AudioFormat,
    AudioPlaybackOutcome,
)
from henry_speech.audio.adapters.avfaudio.driver import AVFAudioDriver
from henry_speech.audio.adapters.avfaudio.process import AVFAudioProcess
from henry_speech.audio.adapters.avfaudio.protocol import (
    AudioDevicesPacket,
    AudioPacket,
    MessageKind,
    PlaybackStatus,
    ProtocolHandshake,
    WireFrame,
)


def test_wire_protocol_round_trip_and_validation() -> None:
    samples = np.asarray([0.25, -0.5], dtype=np.float32).tobytes()
    packet = AudioPacket(16_000, 1, samples)
    audio = packet.encode()
    encoded = WireFrame(MessageKind.PLAY, request_id=42, payload=audio).encode()
    decoded = WireFrame.read_from(io.BytesIO(encoded))

    assert decoded == WireFrame(MessageKind.PLAY, request_id=42, payload=audio)
    assert AudioPacket.decode(audio) == packet
    assert WireFrame.read_from(io.BytesIO()) is None
    assert ProtocolHandshake.decode(ProtocolHandshake.current().encode()).version == 3
    assert (
        PlaybackStatus.decode(PlaybackStatus.PLAYED.encode()) is PlaybackStatus.PLAYED
    )
    devices = AudioDevices(
        input=AudioDevice("MacBook Microphone", "input-1"),
        output=AudioDevice("Studio Display", "output-1"),
    )
    assert AudioDevicesPacket.decode(AudioDevicesPacket(devices).encode()).devices == (
        devices
    )

    with pytest.raises(ValueError, match="positive"):
        AudioPacket(0, 1, samples)
    with pytest.raises(ValueError, match="positive"):
        AudioPacket(16_000, 0, samples)
    with pytest.raises(ValueError, match="frame-aligned"):
        AudioPacket(16_000, 2, b"1234")
    with pytest.raises(RuntimeError, match="truncated"):
        AudioPacket.decode(b"")
    with pytest.raises(EOFError, match="mid-frame"):
        WireFrame.read_from(io.BytesIO(encoded[:-1]))


def test_wire_protocol_retries_partial_writes() -> None:
    class PartialWriter:
        def __init__(self) -> None:
            self.data = bytearray()
            self.flushed = False

        def write(self, data: bytes) -> int:
            chunk = data[:3]
            self.data.extend(chunk)
            return len(chunk)

        def flush(self) -> None:
            self.flushed = True

    frame = WireFrame(MessageKind.DIAGNOSTIC, payload=b"audio ready")
    writer = PartialWriter()

    frame.write_to(writer)

    assert bytes(writer.data) == frame.encode()
    assert writer.flushed


def test_avfaudio_driver_ports_share_process() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.calls: list[object] = []
            self.frame = AudioFormat(44_100, 1).build_frame(
                np.full(4_410, 0.1, dtype=np.float32)
            )

        def open(self) -> None:
            self.calls.append("open")

        def close(self) -> None:
            self.calls.append("close")

        @property
        def devices(self) -> AudioDevices:
            return AudioDevices(
                input=AudioDevice("Fake input", "input-1"),
                output=AudioDevice("Fake output", "output-1"),
            )

        def read(self):
            self.calls.append("read")
            return self.frame

        def play(self, frame):
            self.calls.append(("play", frame))
            return PlaybackStatus.PLAYED

        def interrupt(self) -> None:
            self.calls.append("interrupt")

        def duck(self) -> None:
            self.calls.append("duck")

        def restore(self) -> None:
            self.calls.append("restore")

    process = FakeProcess()
    driver = AVFAudioDriver(process)  # type: ignore[arg-type]
    with driver:
        assert driver.devices == process.devices
        input = driver.input
        output = driver.output
        with input, output:
            captured = input.read()
            assert captured.format == AudioFormat(16_000, 1)
            assert captured.samples_count == 512
            assert output.write(process.frame) is AudioPlaybackOutcome.PLAYED
            output.duck()
            output.restore()
            output.interrupt()

    assert process.calls == [
        "open",
        "read",
        ("play", process.frame),
        "duck",
        "restore",
        "interrupt",
        "close",
    ]


def test_avfaudio_process_lifecycle() -> None:
    helper = Path(__file__).parents[1] / "fixtures" / "fake_avfaudio_helper.py"
    process = AVFAudioProcess((sys.executable, str(helper)))
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="DEBUG")

    try:
        with process:
            assert process._process is not None
            assert os.getpgid(process._process.pid) != os.getpgrp()
            assert process.devices == AudioDevices(
                input=AudioDevice("Fake input", "input-1"),
                output=AudioDevice("Fake output", "output-1"),
            )
            captured = process.read()
            assert captured.format == AudioFormat(48_000, 1)
            np.testing.assert_array_equal(captured.samples, [0.25])
            assert process.play(captured) == PlaybackStatus.PLAYED
            process.duck()
            process.restore()
            process.interrupt()
    finally:
        logger.remove(sink)

    assert any(
        "input_device='Fake input', output_device='Fake output'" in message
        for message in messages
    )
