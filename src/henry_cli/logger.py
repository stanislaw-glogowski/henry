import sys

from loguru import logger


def _ensure_component(record: dict) -> bool:
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
