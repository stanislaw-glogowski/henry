import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import yaml

from henry_conversation import (
    ConversationMessage,
    ConversationProfile,
    ConversationRole,
    ConversationSettings,
    LanguageModelRequest,
    LanguageModelRole,
)
from henry_conversation.graph import ResponseRouter
from henry_conversation.model import LanguageModelService, get_language_model


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    category: str
    text: str
    expected_mode: str


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    id: str
    category: str
    run_kind: str
    expected_mode: str
    selected_mode: str
    first_chunk_seconds: float
    total_seconds: float
    output_characters: int
    response: str


def load_cases(path: Path) -> tuple[BenchmarkCase, ...]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(BenchmarkCase(**item) for item in data["cases"])


async def run_benchmark(
    profile: ConversationProfile,
    settings: ConversationSettings,
    cases: tuple[BenchmarkCase, ...],
) -> tuple[BenchmarkResult, ...]:
    router = ResponseRouter()
    results: list[BenchmarkResult] = []
    adapter = get_language_model(
        profile,
        settings.language_model,
        require_classifier=settings.classify_ambiguous,
    )
    async with LanguageModelService(adapter) as service:
        for index, case in enumerate(cases):
            plan = router.plan(case.text)
            started = perf_counter()
            if settings.classify_ambiguous and router.is_ambiguous(case.text):
                classification = ""
                async for chunk in service.generate(
                    LanguageModelRequest(
                        LanguageModelRole.CLASSIFIER,
                        (
                            ConversationMessage(
                                ConversationRole.SYSTEM,
                                router.CLASSIFICATION_PROMPT,
                            ),
                            ConversationMessage(ConversationRole.USER, case.text),
                        ),
                    )
                ):
                    classification += chunk.content
                plan = router.classified_plan(classification)
            role = LanguageModelRole(plan.mode.value)
            request = LanguageModelRequest(
                role,
                (
                    ConversationMessage(
                        ConversationRole.SYSTEM,
                        profile.prompts.system.format(
                            conversation_summary="No previous conversation."
                        ),
                    ),
                    ConversationMessage(ConversationRole.USER, case.text),
                ),
            )
            first_chunk_seconds: float | None = None
            response = ""
            async for chunk in service.generate(request):
                if first_chunk_seconds is None:
                    first_chunk_seconds = perf_counter() - started
                response += chunk.content
            total_seconds = perf_counter() - started
            results.append(
                BenchmarkResult(
                    id=case.id,
                    category=case.category,
                    run_kind="cold" if index == 0 else "warm",
                    expected_mode=case.expected_mode,
                    selected_mode=plan.mode.value,
                    first_chunk_seconds=first_chunk_seconds or total_seconds,
                    total_seconds=total_seconds,
                    output_characters=len(response),
                    response=response,
                )
            )
    return tuple(results)


def write_report(
    output_directory: Path,
    results: tuple[BenchmarkResult, ...],
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "results.json"
    markdown_path = output_directory / "report.md"
    json_path.write_text(
        json.dumps(
            [asdict(result) for result in results], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    routing_accuracy = (
        sum(result.expected_mode == result.selected_mode for result in results)
        / len(results)
        if results
        else 0.0
    )
    lines = [
        "# Conversation benchmark report",
        "",
        f"- Cases: {len(results)}",
        f"- Routing accuracy: {routing_accuracy:.1%}",
        "",
        "| Case | Run | Expected | Selected | First chunk | Total | Characters |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {result.id} | {result.run_kind} | {result.expected_mode} | "
        f"{result.selected_mode} | "
        f"{result.first_chunk_seconds:.3f} s | {result.total_seconds:.3f} s | "
        f"{result.output_characters} |"
        for result in results
    )
    lines.extend(
        [
            "",
            "## Human review",
            "",
            "Score each spoken response from 1 to 5.",
            "",
            "| Case | Correctness | Naturalness | Brevity | Personality | Notes |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
            *(f"| {result.id} |  |  |  |  |  |" for result in results),
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
