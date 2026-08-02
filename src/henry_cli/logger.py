import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record


def _ensure_component(record: Record) -> bool:
    record["extra"].setdefault("component", "Henry")
    return True


def configure_console_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[component]}</cyan> | <level>{message}</level>"
        ),
        filter=_ensure_component,
    )
