import asyncio
import json
from pathlib import Path

import henry_conversation.model as model_module
from henry_conversation.config import (
    ConversationProfile,
    ConversationPrompts,
    ConversationSettings,
    LanguageModelProfile,
    LanguageModelsProfile,
)
from tests.support import FakeLanguageModel
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
        models=LanguageModelsProfile(
            fast=LanguageModelProfile(langchain="test:fast"),
            detailed=LanguageModelProfile(langchain="test:detailed"),
        ),
        prompts=ConversationPrompts(
            system="System {conversation_summary}",
            opening="Opening {conversation_summary} {recent_conversation}",
            summary="Summary {conversation_summary} {recent_conversation}",
        ),
    )
    monkeypatch.setattr(
        model_module,
        "get_language_model",
        lambda *_: FakeLanguageModel("Answer."),
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
