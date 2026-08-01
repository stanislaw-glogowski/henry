from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import unicodedata
import wave
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from platformdirs import user_data_dir

from henry_speech.audio import AudioFormat, AudioFrame

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_FORMAT = AudioFormat(sample_rate=16_000, channels=1)


@dataclass(frozen=True, slots=True)
class BenchmarkPrompt:
    id: str
    category: str
    text: str
    instruction: str = ""


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    id: str
    description: str
    prompts: tuple[BenchmarkPrompt, ...]


@dataclass(frozen=True, slots=True)
class Recording:
    sample_id: str
    speaker_id: str
    suite: str
    condition: str
    reference_text: str
    wav_path: str
    sample_rate: int
    channels: int
    capture_driver: str
    duration_seconds: float
    sha256: str
    recorded_at: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def suite_directory() -> Path:
    return repository_root() / "benchmarks" / "voice" / "suites"


def benchmark_root(override: Path | None = None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    if value := os.getenv("HENRY_HOME"):
        henry_home = Path(value).expanduser()
    else:
        start = Path.cwd()
        henry_home = next(
            (
                path / ".henry"
                for path in (start, *start.parents)
                if (path / ".henry").is_dir()
            ),
            Path(user_data_dir("Henry")),
        )
    return henry_home / "benchmarks" / "voice"


def validate_identifier(value: str, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{label} must contain lowercase letters, digits, hyphens, "
            f"or underscores; got {value!r}"
        )
    return value


def load_suite(suite_id: str) -> BenchmarkSuite:
    validate_identifier(suite_id, "suite")
    path = suite_directory() / f"{suite_id}.yml"
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark suite does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = tuple(BenchmarkPrompt(**item) for item in data["prompts"])
    ids = [prompt.id for prompt in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Benchmark suite contains duplicate prompt ids: {path}")
    return BenchmarkSuite(
        id=data["id"],
        description=data["description"],
        prompts=prompts,
    )


def timestamp_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz")


def write_wav(path: Path, frame: AudioFrame) -> None:
    if frame.format.channels != 1:
        raise ValueError(
            f"Benchmark WAV output must be mono; got {frame.format.channels} channels"
        )
    samples = np.clip(frame.samples, -1.0, 1.0)
    pcm = np.rint(samples * np.iinfo(np.int16).max).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(frame.format.channels)
        output.setsampwidth(2)
        output.setframerate(frame.format.sample_rate)
        output.writeframes(pcm.tobytes())


def read_wav(path: Path) -> AudioFrame:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        sample_width = source.getsampwidth()
        payload = source.readframes(source.getnframes())
    if sample_width != 2:
        raise ValueError(f"Benchmark WAV must use 16-bit PCM: {path}")
    if channels != 1 or sample_rate != 16_000:
        raise ValueError(f"Benchmark WAV must use 16 kHz mono audio: {path}")
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32)
    samples /= np.iinfo(np.int16).max
    return _FORMAT.build_frame(samples)


def append_recording(path: Path, recording: Recording) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(asdict(recording), ensure_ascii=False) + "\n")


def load_recordings(session_path: Path) -> tuple[Recording, ...]:
    manifest = session_path / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Recording manifest does not exist: {manifest}")
    return tuple(
        Recording(**json.loads(line))
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(
        "".join(
            character if character.isalnum() else " " for character in normalized
        ).split()
    )


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, *, characters: bool = False) -> float:
    expected_text = normalize_text(reference)
    actual_text = normalize_text(hypothesis)
    expected = list(expected_text) if characters else expected_text.split()
    actual = list(actual_text) if characters else actual_text.split()
    if not expected:
        return 0.0 if not actual else 1.0
    return edit_distance(expected, actual) / len(expected)


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as output:
        for row in materialized:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not materialized:
        return
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
