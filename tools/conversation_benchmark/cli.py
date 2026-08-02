import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from henry_resources import LocalStore

from .core import load_cases, run_benchmark, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Henry conversation models")
    parser.add_argument("--profile", default="default")
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("benchmarks/conversation/suites/pl-core.yml"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> tuple[Path, Path]:
    store = LocalStore()
    profile = store.load_profile(args.profile)
    settings = store.load_settings().conversation
    cases = load_cases(args.suite)
    results = await run_benchmark(profile.conversation, settings, cases)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or profile.path.parents[1] / "benchmarks" / "conversation" / (
        f"{args.profile}-{settings.model.adapter}-{timestamp}"
    )
    return write_report(output, results)


def main() -> None:
    json_path, markdown_path = asyncio.run(run(parse_args()))
    print(f"Raw results: {json_path}")
    print(f"Report: {markdown_path}")
