from __future__ import annotations

import queue
import subprocess
import threading
from dataclasses import dataclass, field

from henry_common.components import AbstractResource

from ...domain import AudioDevices, AudioFrame
from .helpers import resolve_helper_executable
from .protocol import (
    AudioDevicesPacket,
    AudioPacket,
    BinaryWriter,
    MessageKind,
    PlaybackStatus,
    ProtocolHandshake,
    WireFrame,
)


@dataclass(slots=True)
class _PendingResponse:
    event: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    playback_status: PlaybackStatus | None = None

    def wait(self, timeout: float | None = None) -> bool:
        completed = self.event.wait(timeout)
        if completed and self.error is not None:
            raise self.error
        return completed

    def complete(self, playback_status: PlaybackStatus | None = None) -> None:
        self.playback_status = playback_status
        self.event.set()

    def fail(self, error: BaseException) -> None:
        self.error = error
        self.event.set()


class AVFAudioProcess(AbstractResource):
    """Own the native duplex helper process and its request/response channel."""

    _CAPTURE_CHANNELS = 1
    _CONTROL_TIMEOUT_SECONDS = 5
    _SHUTDOWN_TIMEOUT_SECONDS = 5
    _START_TIMEOUT_SECONDS = 10

    def __init__(self, command: tuple[str, ...] | None = None) -> None:
        super().__init__()
        self._command = command
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._capture: queue.Queue[AudioFrame | BaseException | None] = queue.Queue()
        self._responses: dict[int, _PendingResponse] = {}
        self._responses_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._ready = threading.Event()
        self._next_request_id = 1
        self._error: BaseException | None = None
        self._devices: AudioDevices | None = None
        self._closing = False
        self._capture_format_logged = False

    def open(self) -> None:
        if self._process is not None:
            raise RuntimeError("Native audio helper process is already open")

        self._reset_runtime_state()
        command = self._command or (str(resolve_helper_executable()),)
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        self._process = process
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(process,),
            name="AVFAudioProcess_reader",
            daemon=True,
        )
        self._reader.start()

        try:
            if not self._ready.wait(self._START_TIMEOUT_SECONDS):
                raise RuntimeError(
                    "Native audio helper did not open within "
                    f"{self._START_TIMEOUT_SECONDS} seconds"
                )
            self._raise_if_failed()
        except BaseException:
            self.close()
            raise

        self._logger.debug("Process OPENED: command={}", command)

    @property
    def devices(self) -> AudioDevices:
        self._require_open()
        devices = self._devices
        if devices is None:
            raise RuntimeError("Native audio helper omitted audio device information")
        return devices

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return

        self._closing = True
        try:
            self._stop_process(process)
        finally:
            self._close_streams(process)
            self._join_reader()
            self._fail_pending(RuntimeError("Native audio helper process is closed"))
            self._capture.put_nowait(None)
            self._ready.clear()
            self._logger.debug("Process CLOSED: returncode={}", process.returncode)

    def read(self) -> AudioFrame:
        self._require_open()
        item = self._capture.get()
        try:
            if item is None:
                raise EOFError("Native audio helper closed the capture stream")
            if isinstance(item, BaseException):
                raise item
            return item
        finally:
            self._capture.task_done()

    def play(self, frame: AudioFrame) -> PlaybackStatus:
        response = self._request(
            MessageKind.PLAY,
            AudioPacket.from_frame(frame).encode(),
        )
        response.wait()
        if response.playback_status is None:
            raise RuntimeError("Native audio helper omitted the playback result")
        return response.playback_status

    def interrupt(self) -> None:
        self._control(MessageKind.STOP, "stop playback")

    def duck(self) -> None:
        self._control(MessageKind.DUCK, "duck playback")

    def restore(self) -> None:
        self._control(MessageKind.RESTORE, "restore playback")

    def _reset_runtime_state(self) -> None:
        self._capture = queue.Queue()
        self._error = None
        self._devices = None
        self._ready.clear()
        self._closing = False
        self._capture_format_logged = False

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            self._write(WireFrame(MessageKind.SHUTDOWN), process.stdin)
        except BrokenPipeError, OSError:
            pass
        try:
            process.wait(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    @staticmethod
    def _close_streams(process: subprocess.Popen[bytes]) -> None:
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()

    def _join_reader(self) -> None:
        reader = self._reader
        self._reader = None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=self._SHUTDOWN_TIMEOUT_SECONDS)

    def _control(self, kind: MessageKind, action: str) -> None:
        response = self._request(kind)
        if not response.wait(self._CONTROL_TIMEOUT_SECONDS):
            raise RuntimeError(
                f"Native audio helper did not {action} within "
                f"{self._CONTROL_TIMEOUT_SECONDS} seconds"
            )

    def _request(self, kind: MessageKind, payload: bytes = b"") -> _PendingResponse:
        process = self._require_open()
        request_id = self._new_request_id()
        response = _PendingResponse()
        with self._responses_lock:
            self._responses[request_id] = response
        try:
            self._write(WireFrame(kind, request_id, payload), process.stdin)
        except BaseException:
            with self._responses_lock:
                self._responses.pop(request_id, None)
            raise
        return response

    def _read_loop(self, process: subprocess.Popen[bytes]) -> None:
        stream = process.stdout
        if stream is None:
            self._set_error(RuntimeError("Native audio helper output stream is closed"))
            self._ready.set()
            return

        try:
            while (frame := WireFrame.read_from(stream)) is not None:
                self._handle(frame)
            if not self._closing:
                raise EOFError(
                    "Native audio helper exited unexpectedly: "
                    f"returncode={process.poll()}"
                )
        except BaseException as error:
            if not self._closing:
                self._set_error(error)
        finally:
            self._ready.set()

    def _handle(self, frame: WireFrame) -> None:
        match frame.kind:
            case MessageKind.READY:
                ProtocolHandshake.decode(frame.payload).verify()
                if self._devices is None:
                    raise RuntimeError(
                        "Native audio helper became ready without device information"
                    )
                self._ready.set()
            case MessageKind.CAPTURE:
                self._handle_capture(frame.payload)
            case MessageKind.PLAYBACK_FINISHED:
                response = self._pop_response(frame.request_id)
                response.complete(PlaybackStatus.decode(frame.payload))
            case MessageKind.STOPPED | MessageKind.GAIN_CHANGED:
                self._pop_response(frame.request_id).complete()
            case MessageKind.DIAGNOSTIC:
                self._logger.debug(
                    "{}", frame.payload.decode("utf-8", errors="replace")
                )
            case MessageKind.DEVICES:
                self._devices = AudioDevicesPacket.decode(frame.payload).devices
                self._logger.debug(
                    "Session OPENED: input_device='{}', output_device='{}'",
                    self._devices.input.name,
                    self._devices.output.name,
                )
            case MessageKind.ERROR:
                message = frame.payload.decode("utf-8", errors="replace")
                raise RuntimeError(f"Native audio helper reported an error: {message}")
            case _:
                raise RuntimeError(
                    f"Unexpected native audio response: {frame.kind.name}"
                )

    def _handle_capture(self, payload: bytes) -> None:
        packet = AudioPacket.decode(payload)
        if packet.channels != self._CAPTURE_CHANNELS:
            raise RuntimeError(
                "Native capture channel mismatch: "
                f"expected {self._CAPTURE_CHANNELS}, got {packet.channels}"
            )
        if not self._capture_format_logged:
            self._logger.debug(
                "Capture OPENED: sample_rate={}, channels={}",
                packet.sample_rate,
                packet.channels,
            )
            self._capture_format_logged = True
        self._capture.put_nowait(packet.to_frame())

    def _pop_response(self, request_id: int) -> _PendingResponse:
        with self._responses_lock:
            response = self._responses.pop(request_id, None)
        if response is None:
            raise RuntimeError(
                f"Native audio helper returned unknown request id: {request_id}"
            )
        return response

    def _new_request_id(self) -> int:
        with self._responses_lock:
            request_id = self._next_request_id
            self._next_request_id = 1 if request_id == 0xFFFFFFFF else request_id + 1
            return request_id

    def _write(self, frame: WireFrame, stream: BinaryWriter | None) -> None:
        if stream is None:
            raise RuntimeError("Native audio helper command stream is closed")
        with self._write_lock:
            frame.write_to(stream)

    def _set_error(self, error: BaseException) -> None:
        self._error = error
        self._capture.put_nowait(error)
        self._fail_pending(error)

    def _fail_pending(self, error: BaseException) -> None:
        with self._responses_lock:
            responses = list(self._responses.values())
            self._responses.clear()
        for response in responses:
            response.fail(error)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _require_open(self) -> subprocess.Popen[bytes]:
        self._raise_if_failed()
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError("Native audio helper process is not open")
        return process
