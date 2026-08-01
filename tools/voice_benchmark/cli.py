from __future__ import annotations

import argparse
from pathlib import Path

from .core import load_suite
from .endpoint import run_endpoint
from .record import record_session
from .report import build_report
from .review import prepare_tts_review
from .stt import run_stt
from .tts import run_tts


def _path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-benchmark",
        description="Record and evaluate Henry's local Polish voice pipeline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="List prompts in a suite")
    list_parser.add_argument("--suite", default="pl-core")

    record = commands.add_parser("record", help="Record a benchmark session")
    record.add_argument("--suite", default="pl-core")
    record.add_argument("--speaker", required=True)
    record.add_argument("--condition", default="quiet")
    record.add_argument("--driver", choices=("avfaudio", "pyaudio"), default="avfaudio")
    record.add_argument("--prompt")
    record.add_argument(
        "--session",
        help="Session directory name; prompted interactively when omitted",
    )
    record.add_argument("--resume", action="store_true")
    record.add_argument("--max-seconds", type=float, default=60.0)
    record.add_argument("--output", type=_path)

    stt = commands.add_parser("stt", help="Benchmark one STT adapter")
    stt.add_argument("--session", type=_path, required=True)
    stt.add_argument(
        "--adapter",
        choices=("mlx:parakeet-tdt", "mlx:qwen3-asr", "mlx:whisper"),
        required=True,
    )
    stt.add_argument("--model")
    stt.add_argument(
        "--language",
        help="optional adapter-specific language hint, for example 'pl' for Whisper",
    )
    stt.add_argument("--output", type=_path)

    endpoint = commands.add_parser("endpoint", help="Benchmark VAD endpointing")
    endpoint.add_argument("--session", type=_path, required=True)
    endpoint.add_argument("--output", type=_path)

    tts = commands.add_parser("tts", help="Benchmark one TTS adapter")
    tts.add_argument("--suite", default="pl-tts")
    tts.add_argument("--adapter", choices=("piper", "mlx:chatterbox"), required=True)
    tts.add_argument("--model", required=True)
    tts.add_argument("--output", type=_path)

    report = commands.add_parser("report", help="Build a Markdown result summary")
    report.add_argument("--results", type=_path, required=True)

    review = commands.add_parser("tts-review", help="Prepare a blind TTS comparison")
    review.add_argument("--results", type=_path, nargs="+", required=True)
    review.add_argument("--output", type=_path, required=True)
    review.add_argument("--seed", default="henry")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    match args.command:
        case "list":
            suite = load_suite(args.suite)
            print(f"{suite.id}: {suite.description}")
            for prompt in suite.prompts:
                print(f"\n{prompt.id} [{prompt.category}]\n{prompt.text}")
                if prompt.instruction:
                    print(f"Instruction: {prompt.instruction}")
        case "record":
            print(f"Session saved to: {record_session(args)}")
        case "stt":
            print(f"Results saved to: {run_stt(args)}")
        case "endpoint":
            print(f"Results saved to: {run_endpoint(args)}")
        case "tts":
            print(f"Results saved to: {run_tts(args)}")
        case "report":
            print(f"Report saved to: {build_report(args.results)}")
        case "tts-review":
            print(
                "Listening review saved to: "
                f"{prepare_tts_review(args.results, args.output, args.seed)}"
            )
    return 0
