from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from henry_speech.audio import AudioBuffer, AudioFrame
from henry_speech.audio.adapters import get_audio_driver
from henry_speech.audio.config import AudioSettings

from .core import (
    BenchmarkPrompt,
    Recording,
    append_recording,
    benchmark_root,
    file_sha256,
    load_recordings,
    load_suite,
    timestamp_id,
    validate_identifier,
    write_wav,
)

_METADATA_TEMPLATE = """# Complete this file before evaluating the session.
# Do not add names, dates of birth, or other direct personal identifiers.
speaker:
  age_group: ""
  native_language: ""
  regional_accent: ""
  speech_notes: ""
recording:
  device: ""
  microphone: ""
  microphone_distance_cm:
  room: ""
  background_noise: ""
consent:
  participant_or_guardian_confirmed: false
notes: ""
"""


def _capture(audio_input, max_seconds: float) -> AudioFrame:
    stopped = threading.Event()

    def wait_for_enter() -> None:
        input()
        stopped.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()
    buffer = AudioBuffer()
    samples_limit = int(16_000 * max_seconds)
    samples = 0
    while not stopped.is_set() and samples < samples_limit:
        frame = audio_input.read()
        buffer.append(frame)
        samples += frame.samples_count
    captured = buffer.build()
    if captured is None:
        raise RuntimeError("Recording ended before any audio frame was captured")
    return captured


def _show_prompt(prompt: BenchmarkPrompt, index: int, total: int) -> None:
    print(f"\n[{index}/{total}] {prompt.id} · {prompt.category}")
    if prompt.instruction:
        print(f"Instruction: {prompt.instruction}")
    print("\n" + prompt.text + "\n")


def _session_name(requested: str | None) -> str:
    if requested is None:
        requested = input(
            "Session directory name (leave blank to use the current time): "
        ).strip()
    return validate_identifier(requested or timestamp_id(), "session")


def _create_session_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Recording session already exists: {path}. "
            "Choose another name or use --resume --session NAME."
        ) from error
    _ensure_metadata_template(path)


def _ensure_metadata_template(path: Path) -> None:
    metadata = path / "metadata.yml"
    if not metadata.exists():
        metadata.write_text(_METADATA_TEMPLATE, encoding="utf-8")


def record_session(args) -> Path:
    suite = load_suite(args.suite)
    speaker = validate_identifier(args.speaker, "speaker")
    condition = validate_identifier(args.condition, "condition")
    root = benchmark_root(args.output)
    speaker_root = root / "recordings" / suite.id / speaker
    if args.resume:
        if args.session:
            session_path = speaker_root / validate_identifier(args.session, "session")
            if not session_path.is_dir():
                raise FileNotFoundError(
                    f"Recording session cannot be resumed: {session_path}"
                )
        else:
            sessions = sorted(path for path in speaker_root.glob("*") if path.is_dir())
            if not sessions:
                raise FileNotFoundError(
                    f"No recording session can be resumed below: {speaker_root}"
                )
            session_path = max(sessions, key=lambda path: path.stat().st_mtime_ns)
        _ensure_metadata_template(session_path)
    else:
        session_path = speaker_root / _session_name(args.session)
        _create_session_directory(session_path)
    manifest_path = session_path / "manifest.jsonl"
    completed = (
        {item.sample_id for item in load_recordings(session_path)}
        if manifest_path.exists()
        else set()
    )
    prompts = tuple(
        prompt
        for prompt in suite.prompts
        if prompt.id not in completed
        and (args.prompt is None or prompt.id == args.prompt)
    )
    if args.prompt is not None and not prompts and args.prompt not in completed:
        raise ValueError(f"Unknown prompt id for suite {suite.id!r}: {args.prompt!r}")

    print(f"Suite: {suite.id} — {suite.description}")
    print(f"Session: {session_path}")
    print(f"Session metadata: {session_path / 'metadata.yml'}")
    print("Enter starts and stops recording. Choose an action after each take.")
    with get_audio_driver(AudioSettings(driver=args.driver)) as driver:
        audio_input = driver.get_input()
        audio_output = driver.get_output()
        for index, prompt in enumerate(prompts, start=1):
            while True:
                _show_prompt(prompt, index, len(prompts))
                input("Read the text silently and prepare. Press Enter when ready…")
                print("Recording… Press Enter to stop.")
                frame = _capture(audio_input, args.max_seconds)
                temporary = session_path / f".{prompt.id}.wav"
                write_wav(temporary, frame)
                while True:
                    action = (
                        input("[a] accept, [p] play, [r] retake, [s] skip, [q] quit: ")
                        .strip()
                        .lower()
                    )
                    if action == "p":
                        audio_output.write(frame)
                        continue
                    if action in {"", "a", "r", "s", "q"}:
                        break
                    print("Unknown action; the recording still needs a decision.")
                if action == "r":
                    temporary.unlink(missing_ok=True)
                    continue
                if action == "s":
                    temporary.unlink(missing_ok=True)
                    break
                if action == "q":
                    temporary.unlink(missing_ok=True)
                    return session_path

                target = session_path / f"{prompt.id}.wav"
                temporary.replace(target)
                append_recording(
                    manifest_path,
                    Recording(
                        sample_id=prompt.id,
                        speaker_id=speaker,
                        suite=suite.id,
                        condition=condition,
                        reference_text=prompt.text,
                        wav_path=target.name,
                        sample_rate=frame.format.sample_rate,
                        channels=frame.format.channels,
                        capture_driver=args.driver,
                        duration_seconds=frame.samples_count / frame.format.sample_rate,
                        sha256=file_sha256(target),
                        recorded_at=datetime.now(UTC).isoformat(),
                    ),
                )
                break
    return session_path
