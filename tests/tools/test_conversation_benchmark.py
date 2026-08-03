import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import henry_conversation.model as model_module
from henry_conversation.config import ConversationSettings
from henry_conversation.profile import ConversationProfile, ConversationPrompts
from tests.support import FakeLanguageModel
from tools.conversation_benchmark import cli as cli_module
from tools.conversation_benchmark.cli import override_model_id
from tools.conversation_benchmark.core import (
    BenchmarkCase,
    load_cases,
    run_benchmark,
    write_report,
)


def test_conversation_benchmark_loads_runs_and_writes_reports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite.yml"
    suite.write_text(
        """
cases:
  - id: fast-001
    category: factual
    text: Krótkie pytanie?
    expected_mode: fast
""".strip(),
        encoding="utf-8",
    )
    cases = load_cases(suite)
    assert cases == (BenchmarkCase("fast-001", "factual", "Krótkie pytanie?", "fast"),)

    profile = ConversationProfile(
        models={
            "fast": {"model_id": "test:fast"},
            "detailed": {"model_id": "test:detailed"},
        },
        prompts=ConversationPrompts(
            system="System {conversation_summary}",
            opening="Opening {conversation_summary} {recent_conversation}",
            summary="Summary {conversation_summary} {recent_conversation}",
        ),
    )
    monkeypatch.setattr(
        model_module,
        "get_language_model",
        lambda *_, **__: FakeLanguageModel("Answer."),
    )
    monkeypatch.setattr(
        "tools.conversation_benchmark.core.get_language_model",
        model_module.get_language_model,
    )
    results = asyncio.run(run_benchmark(profile, ConversationSettings(), cases))
    assert results[0].selected_mode == "fast"
    assert results[0].response == "Answer."

    json_path, markdown_path = write_report(tmp_path / "result", results)
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["id"] == "fast-001"
    report = markdown_path.read_text(encoding="utf-8")
    assert "Routing accuracy: 100.0%" in report
    assert "fast-001" in report


def test_conversation_benchmark_writes_empty_report(tmp_path: Path) -> None:
    _, markdown_path = write_report(tmp_path, ())
    assert "Routing accuracy: 0.0%" in markdown_path.read_text(encoding="utf-8")


def test_conversation_benchmark_overrides_every_model_role() -> None:
    profile = ConversationProfile(
        models={
            "fast": {"model_id": "fast"},
            "detailed": {"model_id": "detailed"},
            "classifier": {"model_id": "classifier"},
        },
        prompts=ConversationPrompts(
            system="System",
            opening="Opening",
            summary="Summary",
        ),
    )

    unchanged = override_model_id(profile, None)
    overridden = override_model_id(profile, "candidate")

    assert unchanged is profile
    assert {
        role: options["model_id"] for role, options in overridden.models.items()
    } == {
        "fast": "candidate",
        "detailed": "candidate",
        "classifier": "candidate",
    }


def test_conversation_benchmark_cli_uses_configured_adapter(
    monkeypatch,
    tmp_path: Path,
) -> None:
    profile = SimpleNamespace(
        conversation=object(),
        path=tmp_path / "profiles" / "default",
    )
    settings = ConversationSettings()
    store = SimpleNamespace(
        load_profile=lambda _: profile,
        load_settings=lambda: SimpleNamespace(conversation=settings),
    )
    output_directories: list[Path] = []

    async def fake_run_benchmark(*args):
        assert args == (profile.conversation, settings, ())
        return ()

    def fake_write_report(output: Path, results):
        assert results == ()
        output_directories.append(output)
        return output / "results.json", output / "report.md"

    monkeypatch.setattr(cli_module, "LocalStore", lambda: store)
    monkeypatch.setattr(cli_module, "load_cases", lambda _: ())
    monkeypatch.setattr(cli_module, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(cli_module, "write_report", fake_write_report)

    asyncio.run(
        cli_module.run(
            SimpleNamespace(
                profile="default",
                model_id=None,
                suite=tmp_path / "suite.yml",
                output=None,
            )
        )
    )

    assert output_directories[0].name.startswith("default-mlx-")
