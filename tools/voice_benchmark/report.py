from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_results(root: Path, name: str) -> list[dict]:
    paths = [root / name, *sorted(root.glob(f"*/{name}"))]
    return [row for path in paths for row in _load(path)]


def _groups(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["adapter"], row["model"]), []).append(row)
    return groups


def _average(rows: list[dict], key: str) -> str:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return f"{mean(values):.3f}" if values else "n/a"


def build_report(result_path: Path) -> Path:
    result_path = result_path.expanduser().resolve()
    stt = _load_results(result_path, "stt.jsonl")
    endpoint = _load_results(result_path, "endpoint.jsonl")
    tts = _load_results(result_path, "tts.jsonl")
    lines = ["# Henry voice benchmark", "", f"Result directory: `{result_path}`", ""]
    if stt:
        lines.extend(
            [
                "## Speech recognition",
                "",
                "| Adapter | Model | Samples | Mean WER | Mean CER | Mean RTF |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for (adapter, model), rows in _groups(stt).items():
            lines.append(
                f"| {adapter} | {model} | {len(rows)} | "
                f"{_average(rows, 'wer')} | {_average(rows, 'cer')} | "
                f"{_average(rows, 'real_time_factor')} |"
            )
        lines.append("")
    if endpoint:
        latencies = [
            float(row["endpoint_latency_ms"])
            for row in endpoint
            if row.get("endpoint_latency_ms") not in {None, ""}
        ]
        premature = sum(bool(row["premature_endpoint"]) for row in endpoint)
        lines.extend(
            [
                "## Endpointing",
                "",
                f"- Samples: {len(endpoint)}",
                f"- Median endpoint latency: {median(latencies):.1f} ms"
                if latencies
                else "- Median endpoint latency: n/a",
                f"- Premature endpoints: {premature}",
                "",
            ]
        )
    if tts:
        lines.extend(
            [
                "## Speech synthesis",
                "",
                "| Adapter | Model | Samples | Mean first audio | Mean RTF |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for (adapter, model), rows in _groups(tts).items():
            lines.append(
                f"| {adapter} | {model} | {len(rows)} | "
                f"{_average(rows, 'first_audio_seconds')} s | "
                f"{_average(rows, 'real_time_factor')} |"
            )
        lines.extend(
            ["", "Naturalness and pronunciation require a blind listening review.", ""]
        )
    if not stt and not endpoint and not tts:
        lines.append("No benchmark result files were found.")
    report = result_path / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
