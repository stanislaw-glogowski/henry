import inspect

from henry_cli.main import main as cli_main
from henry_debugger.main import main as debugger_main


def test_console_entrypoints_are_synchronous() -> None:
    assert not inspect.iscoroutinefunction(cli_main)
    assert not inspect.iscoroutinefunction(debugger_main)
