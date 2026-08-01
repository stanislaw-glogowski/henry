import json
from pathlib import Path

import numpy as np
import pytest

from henry_speech.audio import AudioFormat
from tools.voice_benchmark.cli import build_parser
from tools.voice_benchmark.core import (
    error_rate,
    load_suite,
    normalize_text,
    read_wav,
    validate_identifier,
    write_rows,
    write_wav,
)
from tools.voice_benchmark.record import (
    _create_session_directory,
    _session_name,
)
from tools.voice_benchmark.report import build_report
from tools.voice_benchmark.review import prepare_tts_review


def test_suite_and_cli_contract() -> None:
    suite = load_suite("pl-core")
    assert suite.id == "pl-core"
    assert len(suite.prompts) >= 20
    assert build_parser().parse_args(["list", "--suite", "pl-tts"]).command == "list"
    stt = build_parser().parse_args(
        [
            "stt",
            "--session",
            "session",
            "--adapter",
            "mlx:whisper",
            "--language",
            "pl",
        ]
    )
    assert stt.language == "pl"
    with pytest.raises(ValueError, match="lowercase"):
        validate_identifier("Speaker One", "speaker")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_suite("missing")


def test_recording_session_name_and_metadata_template(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "family-test")
    assert _session_name(None) == "family-test"

    session = tmp_path / "family-test"
    _create_session_directory(session)
    metadata = (session / "metadata.yml").read_text(encoding="utf-8")
    assert 'age_group: ""' in metadata
    assert "participant_or_guardian_confirmed: false" in metadata
    assert not (session / "manifest.jsonl").exists()

    with pytest.raises(FileExistsError, match="already exists"):
        _create_session_directory(session)


def test_wav_and_text_metrics(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    samples = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    write_wav(path, AudioFormat(16_000, 1).build_frame(samples))
    restored = read_wav(path)
    np.testing.assert_allclose(restored.samples, samples, atol=1 / 32767)
    assert normalize_text("  ŻÓŁĆ, AsyncIO! ") == "żółć asyncio"
    assert error_rate("jeden dwa", "jeden trzy") == 0.5
    assert error_rate("abc", "adc", characters=True) == pytest.approx(1 / 3)
    assert error_rate("", "") == 0
    assert error_rate("", "tekst") == 1


def test_write_rows_and_recursive_report(tmp_path: Path) -> None:
    first = tmp_path / "parakeet"
    second = tmp_path / "qwen"
    row = {
        "sample_id": "one",
        "speaker_id": "speaker-01",
        "condition": "quiet",
        "adapter": "mlx:parakeet-tdt",
        "model": "parakeet",
        "reference": "tekst",
        "hypothesis": "tekst",
        "wer": 0.0,
        "cer": 0.0,
        "model_load_seconds": 1.0,
        "inference_seconds": 0.1,
        "audio_seconds": 1.0,
        "real_time_factor": 0.1,
    }
    write_rows(first / "stt", [row])
    write_rows(
        second / "stt",
        [{**row, "adapter": "mlx:qwen3-asr", "model": "qwen", "wer": 0.2}],
    )
    report = build_report(tmp_path)
    content = report.read_text(encoding="utf-8")
    assert "mlx:parakeet-tdt" in content
    assert "mlx:qwen3-asr" in content


def test_blind_tts_review(tmp_path: Path) -> None:
    result_paths = [tmp_path / "piper", tmp_path / "chatterbox"]
    for index, result_path in enumerate(result_paths):
        audio = result_path / "audio"
        audio.mkdir(parents=True)
        (audio / "tts-short-001.wav").write_bytes(bytes([index]))
        row = {
            "sample_id": "tts-short-001",
            "adapter": f"adapter-{index}",
            "model": f"model-{index}",
            "wav_path": "audio/tts-short-001.wav",
        }
        (result_path / "tts.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    review = prepare_tts_review(result_paths, tmp_path / "review", "seed")
    assert sorted(path.name for path in (review / "audio").iterdir()) == [
        "tts-short-001-a.wav",
        "tts-short-001-b.wav",
    ]
    assert "naturalness_1_5" in (review / "ratings.csv").read_text()
    mapping = json.loads((review / "mapping.json").read_text())
    assert {item["adapter"] for item in mapping} == {"adapter-0", "adapter-1"}

    with pytest.raises(ValueError, match="at least two"):
        prepare_tts_review(result_paths[:1], tmp_path / "invalid", "seed")
