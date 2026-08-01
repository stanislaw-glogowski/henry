from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path


def _read_results(path: Path) -> dict[str, dict]:
    manifest = path / "tts.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"TTS benchmark result does not exist: {manifest}")
    return {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def prepare_tts_review(
    result_paths: list[Path],
    output: Path,
    seed: str,
) -> Path:
    if len(result_paths) < 2:
        raise ValueError("A blind TTS review requires at least two result directories")
    candidates = [
        (path.resolve(), _read_results(path.resolve())) for path in result_paths
    ]
    sample_ids = set(candidates[0][1])
    if any(set(rows) != sample_ids for _, rows in candidates[1:]):
        raise ValueError("TTS result directories must contain identical sample ids")

    output = output.expanduser().resolve()
    audio_output = output / "audio"
    audio_output.mkdir(parents=True, exist_ok=True)
    randomizer = random.Random(seed)
    mapping: list[dict] = []
    ratings: list[dict] = []
    for sample_id in sorted(sample_ids):
        order = list(range(len(candidates)))
        randomizer.shuffle(order)
        for position, candidate_index in enumerate(order):
            label = chr(ord("a") + position)
            root, rows = candidates[candidate_index]
            row = rows[sample_id]
            source = root / row["wav_path"]
            target = audio_output / f"{sample_id}-{label}.wav"
            shutil.copy2(source, target)
            mapping.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "adapter": row["adapter"],
                    "model": row["model"],
                    "source": str(source),
                }
            )
            ratings.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "naturalness_1_5": "",
                    "intelligibility_1_5": "",
                    "pronunciation_1_5": "",
                    "prosody_1_5": "",
                    "notes": "",
                }
            )
    (output / "mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "ratings.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ratings[0]))
        writer.writeheader()
        writer.writerows(ratings)
    (output / "README.md").write_text(
        "# Blind TTS review\n\n"
        "Listen to files in `audio/` in sample order and fill in `ratings.csv`. "
        "Do not open `mapping.json` before completing the review.\n",
        encoding="utf-8",
    )
    return output
