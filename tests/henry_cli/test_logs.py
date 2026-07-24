from loguru import logger

from henry_cli.logs import LogBuffer, configure_console_logger


def test_log_buffer_formats_logs_and_discards_oldest_lines() -> None:
    buffer = LogBuffer(level="INFO", max_lines=2)

    try:
        logger.bind(component="Test").info("first")
        logger.bind(component="Test").warning("second")
        logger.bind(component="Test").error("third")

        lines = buffer.drain()

        assert len(lines) == 2
        assert "second" in lines[0]
        assert "third" in lines[1]
        assert buffer.drain() == []
    finally:
        logger.remove()


def test_log_buffer_strips_trailing_whitespace() -> None:
    buffer = LogBuffer()

    try:
        buffer.write("message \n")

        assert buffer.drain() == ["message"]
    finally:
        logger.remove()


def test_configure_console_logger_writes_component_and_message(capsys) -> None:
    configure_console_logger("DEBUG")

    try:
        logger.bind(component="Test").debug("visible")

        output = capsys.readouterr().out
        assert "@Test" in output
        assert "visible" in output
    finally:
        logger.remove()
