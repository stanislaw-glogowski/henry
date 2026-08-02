from collections import deque
from threading import Lock

from loguru import logger

from .logger import _ensure_component


class LogBuffer:
    def __init__(self, max_lines: int = 1_000) -> None:
        if max_lines <= 0:
            raise ValueError(f"max_lines must be positive; got {max_lines}")
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._lock = Lock()

    def write(self, message: str) -> None:
        line = message.rstrip()
        if not line:
            return
        with self._lock:
            self._lines.append(line)

    def drain(self) -> tuple[str, ...]:
        with self._lock:
            lines = tuple(self._lines)
            self._lines.clear()
        return lines


def configure_ui_logger(buffer: LogBuffer, level: str = "DEBUG") -> None:
    logger.remove()
    logger.add(
        buffer.write,
        level=level,
        colorize=False,
        format=("{time:HH:mm:ss.SSS}  {level: <8}  {extra[component]: <24}  {message}"),
        filter=_ensure_component,
    )
