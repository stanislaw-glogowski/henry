import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Protocol

from ...domain import AudioDevice, AudioDevices, AudioFormat, AudioFrame


class BinaryReader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


class BinaryWriter(Protocol):
    def write(self, data: bytes, /) -> int | None: ...

    def flush(self) -> object: ...


class MessageKind(IntEnum):
    PLAY = 1
    STOP = 2
    SHUTDOWN = 3
    DUCK = 4
    RESTORE = 5

    READY = 128
    CAPTURE = 129
    PLAYBACK_FINISHED = 130
    STOPPED = 131
    GAIN_CHANGED = 132
    DIAGNOSTIC = 133
    DEVICES = 134
    ERROR = 255


class PlaybackStatus(IntEnum):
    PLAYED = 0
    INTERRUPTED = 1

    @classmethod
    def decode(cls, payload: bytes) -> PlaybackStatus:
        if len(payload) != 1:
            raise RuntimeError(
                "Native audio helper returned an invalid playback result: "
                f"expected 1 byte, got {len(payload)}"
            )
        try:
            return cls(payload[0])
        except ValueError as error:
            raise RuntimeError(
                f"Native audio helper returned an unknown playback result: {payload[0]}"
            ) from error

    def encode(self) -> bytes:
        return bytes([self])


@dataclass(frozen=True, slots=True)
class ProtocolHandshake:
    CURRENT_VERSION: ClassVar[int] = 3

    version: int

    @classmethod
    def current(cls) -> ProtocolHandshake:
        return cls(cls.CURRENT_VERSION)

    @classmethod
    def decode(cls, payload: bytes) -> ProtocolHandshake:
        if len(payload) != 2:
            raise RuntimeError(
                "Native audio helper returned an invalid handshake: "
                f"expected 2 bytes, got {len(payload)}"
            )
        return cls(int.from_bytes(payload, "little"))

    def encode(self) -> bytes:
        return self.version.to_bytes(2, "little")

    def verify(self) -> None:
        if self.version != self.CURRENT_VERSION:
            raise RuntimeError(
                "Unsupported native audio protocol version: "
                f"expected {self.CURRENT_VERSION}, got {self.version}"
            )


@dataclass(frozen=True, slots=True)
class AudioPacket:
    _HEADER: ClassVar[struct.Struct] = struct.Struct("<IH")
    _SAMPLE_WIDTH: ClassVar[int] = 4

    sample_rate: int
    channels: int
    samples: bytes

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive; got {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive; got {self.channels}")
        if len(self.samples) % (self._SAMPLE_WIDTH * self.channels):
            raise ValueError(
                "samples must contain frame-aligned float32 data; "
                f"got {len(self.samples)} bytes for {self.channels} channels"
            )

    @classmethod
    def from_frame(cls, frame: AudioFrame) -> AudioPacket:
        return cls(
            sample_rate=frame.format.sample_rate,
            channels=frame.format.channels,
            samples=frame.to_bytes(),
        )

    @classmethod
    def decode(cls, payload: bytes) -> AudioPacket:
        if len(payload) < cls._HEADER.size:
            raise RuntimeError(
                "Native audio payload header is truncated: "
                f"expected at least {cls._HEADER.size} bytes, got {len(payload)}"
            )
        sample_rate, channels = cls._HEADER.unpack_from(payload)
        try:
            return cls(sample_rate, channels, payload[cls._HEADER.size :])
        except ValueError as error:
            raise RuntimeError(f"Native audio payload is invalid: {error}") from error

    def encode(self) -> bytes:
        return self._HEADER.pack(self.sample_rate, self.channels) + self.samples

    def to_frame(self) -> AudioFrame:
        return AudioFormat(self.sample_rate, self.channels).build_frame(self.samples)


@dataclass(frozen=True, slots=True)
class AudioDevicesPacket:
    devices: AudioDevices

    @classmethod
    def decode(cls, payload: bytes) -> AudioDevicesPacket:
        try:
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise TypeError("root value must be an object")
            devices = AudioDevices(
                input=cls._decode_device(data.get("input"), "input"),
                output=cls._decode_device(data.get("output"), "output"),
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise RuntimeError(
                f"Native audio device payload is invalid: {error}"
            ) from error
        return cls(devices)

    def encode(self) -> bytes:
        return json.dumps(
            {
                "input": self._encode_device(self.devices.input),
                "output": self._encode_device(self.devices.output),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()

    @staticmethod
    def _decode_device(data: object, role: str) -> AudioDevice:
        if not isinstance(data, dict):
            raise TypeError(f"{role} device must be an object")
        name = data.get("name")
        identifier = data.get("identifier")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{role} device name must be a non-empty string")
        if identifier is not None and not isinstance(identifier, str):
            raise TypeError(f"{role} device identifier must be a string or null")
        return AudioDevice(name=name, identifier=identifier)

    @staticmethod
    def _encode_device(device: AudioDevice) -> dict[str, str | None]:
        return {
            "name": device.name,
            "identifier": device.identifier,
        }


@dataclass(frozen=True, slots=True)
class WireFrame:
    _HEADER: ClassVar[struct.Struct] = struct.Struct("<BII")

    kind: MessageKind
    request_id: int = 0
    payload: bytes = b""

    @classmethod
    def read_from(cls, stream: BinaryReader) -> WireFrame | None:
        header = cls._read_exactly(stream, cls._HEADER.size, allow_clean_eof=True)
        if header is None:
            return None

        raw_kind, request_id, payload_size = cls._HEADER.unpack(header)
        try:
            kind = MessageKind(raw_kind)
        except ValueError as error:
            raise RuntimeError(
                f"Unknown native audio message kind: {raw_kind}"
            ) from error

        payload = cls._read_exactly(stream, payload_size, allow_clean_eof=False)
        return cls(kind, request_id, payload or b"")

    def encode(self) -> bytes:
        header = self._HEADER.pack(self.kind, self.request_id, len(self.payload))
        return header + self.payload

    def write_to(self, stream: BinaryWriter) -> None:
        encoded = self.encode()
        offset = 0
        while offset < len(encoded):
            written = stream.write(encoded[offset:])
            if written is None or written <= 0:
                raise BrokenPipeError("Native audio stream did not accept frame data")
            offset += written
        stream.flush()

    @staticmethod
    def _read_exactly(
        stream: BinaryReader,
        size: int,
        *,
        allow_clean_eof: bool,
    ) -> bytes | None:
        data = bytearray()
        while len(data) < size:
            chunk = stream.read(size - len(data))
            if not chunk:
                if allow_clean_eof and not data:
                    return None
                raise EOFError(
                    "Native audio stream ended mid-frame: "
                    f"expected {size} bytes, got {len(data)}"
                )
            data.extend(chunk)
        return bytes(data)
