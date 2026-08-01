import sys

import numpy as np

from henry_speech.audio import AudioDevice, AudioDevices
from henry_speech.audio.adapters.avfaudio.protocol import (
    AudioDevicesPacket,
    AudioPacket,
    MessageKind,
    PlaybackStatus,
    ProtocolHandshake,
    WireFrame,
)


def send(frame: WireFrame) -> None:
    frame.write_to(sys.stdout.buffer)


send(
    WireFrame(
        MessageKind.DEVICES,
        payload=AudioDevicesPacket(
            AudioDevices(
                input=AudioDevice("Fake input", "input-1"),
                output=AudioDevice("Fake output", "output-1"),
            )
        ).encode(),
    )
)
send(
    WireFrame(
        MessageKind.READY,
        payload=ProtocolHandshake.current().encode(),
    )
)
send(
    WireFrame(
        MessageKind.CAPTURE,
        payload=AudioPacket(
            48_000,
            1,
            np.asarray([0.25], dtype=np.float32).tobytes(),
        ).encode(),
    )
)

while (frame := WireFrame.read_from(sys.stdin.buffer)) is not None:
    match frame.kind:
        case MessageKind.PLAY:
            send(
                WireFrame(
                    MessageKind.PLAYBACK_FINISHED,
                    request_id=frame.request_id,
                    payload=PlaybackStatus.PLAYED.encode(),
                )
            )
        case MessageKind.STOP:
            send(WireFrame(MessageKind.STOPPED, request_id=frame.request_id))
        case MessageKind.DUCK | MessageKind.RESTORE:
            send(WireFrame(MessageKind.GAIN_CHANGED, request_id=frame.request_id))
        case MessageKind.SHUTDOWN:
            break
