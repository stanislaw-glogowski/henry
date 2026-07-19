from collections import deque
from threading import Lock

from loguru import logger


class LogBuffer:
    def __init__(self, max_lines: int = 1_000) -> None:
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._lock = Lock()

    def write(self, message: str) -> None:
        line = str(message).rstrip()

        with self._lock:
            self._lines.append(line)

    def drain(self) -> list[str]:
        with self._lock:
            lines = list(self._lines)
            self._lines.clear()

        return lines


def configure_loger(level: str = "TRACE") -> LogBuffer:
    buffer = LogBuffer()

    logger.remove()
    logger.configure(
        extra={"component": "App"},
    )
    logger.add(
        buffer.write,
        level=level,
        format="{time:HH:mm:ss.SSS} | {level: <8} | {extra[component]} | {message}",
    )

    return buffer
