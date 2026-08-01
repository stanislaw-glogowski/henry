# Henry voice benchmark

This benchmark provides repeatable comparisons of Polish STT, endpointing, and
TTS. The text displayed during recording is the reference transcription. Model
output is always stored separately.

Recordings and results are stored below `$HENRY_HOME/benchmarks/voice`, the
nearest `.henry/benchmarks/voice`, or the platform user-data directory. They are
not part of the repository.

## Recording

Display a suite without opening an audio device or loading a model:

```bash
uv run python -m tools.voice_benchmark list --suite pl-core
```

Start a complete session through the native AVFAudio path:

```bash
uv run python -m tools.voice_benchmark record \
  --suite pl-core \
  --speaker speaker-01 \
  --condition quiet
```

The recorder asks for a session directory name. Leave it blank to use a UTC
timestamp. The prompt is then displayed for as long as the participant needs to
prepare. Press Enter to start recording and Enter again to stop. Each take can
be accepted, played, repeated, skipped, or left for a later resumed session.

To avoid the interactive directory-name question, pass a name explicitly:

```bash
uv run python -m tools.voice_benchmark record \
  --suite pl-core \
  --speaker speaker-01 \
  --condition quiet \
  --session kitchen-morning
```

Resume a named session:

```bash
uv run python -m tools.voice_benchmark record \
  --suite pl-core \
  --speaker speaker-01 \
  --condition quiet \
  --session kitchen-morning \
  --resume
```

Without `--session`, `--resume` selects the most recently modified session for
the speaker and suite.

Important options:

| Option | Meaning |
|---|---|
| `--suite` | `pl-core`, `pl-turn-taking`, or `pl-tts` |
| `--speaker` | Anonymous participant identifier |
| `--condition` | For example `quiet`, `noise`, `far-field`, or `headset` |
| `--driver` | `avfaudio`, or `pyaudio` for comparison |
| `--prompt` | Record only one selected prompt |
| `--session` | Session directory name |
| `--resume` | Continue an existing session |
| `--output` | Explicit benchmark data directory |

Each session directory contains:

```text
recordings/pl-core/speaker-01/kitchen-morning/
├── metadata.yml
├── manifest.jsonl
├── pl-short-001.wav
└── pl-short-002.wav
```

`metadata.yml` is a blank, human-editable template for anonymous participant,
room, device, and consent information. `manifest.jsonl` is maintained by the
recorder and describes every accepted WAV, including its reference text, audio
format, duration, capture driver, and SHA-256 checksum.

Files in `suites/` contain the exact Polish text to read. `pl-turn-taking` also
contains pause instructions; instructions are not part of the reference
transcription.

## STT benchmark

Run every adapter against the same session and use a separate output directory:

```bash
uv run python -m tools.voice_benchmark stt \
  --session .henry/benchmarks/voice/recordings/pl-core/speaker-01/SESSION \
  --adapter mlx:parakeet-tdt \
  --output .henry/benchmarks/voice/results/parakeet

uv run python -m tools.voice_benchmark stt \
  --session .henry/benchmarks/voice/recordings/pl-core/speaker-01/SESSION \
  --adapter mlx:qwen3-asr \
  --model mlx-community/Qwen3-ASR-0.6B-8bit \
  --output .henry/benchmarks/voice/results/qwen3

uv run python -m tools.voice_benchmark stt \
  --session .henry/benchmarks/voice/recordings/pl-core/speaker-01/SESSION \
  --adapter mlx:whisper \
  --model mlx-community/whisper-large-v3-turbo-asr-fp16 \
  --language pl \
  --output .henry/benchmarks/voice/results/whisper
```

The runner writes JSONL and CSV containing the transcription, WER, CER,
inference time, and real-time factor. The first model run may download missing
weights.

## Endpointing

```bash
uv run python -m tools.voice_benchmark endpoint \
  --session .henry/benchmarks/voice/recordings/pl-turn-taking/speaker-01/SESSION \
  --output .henry/benchmarks/voice/results/endpoint
```

The command replays each recording as frames through the production VAD and
`UtteranceSegmenter`. Results include endpoint latency, detected utterance
count, and premature endpoint detection.

## TTS benchmark

```bash
uv run python -m tools.voice_benchmark tts \
  --suite pl-tts \
  --adapter piper \
  --model pl/pl_PL/bass/high/pl_PL-bass-high.onnx \
  --output .henry/benchmarks/voice/results/piper

uv run python -m tools.voice_benchmark tts \
  --suite pl-tts \
  --adapter mlx:chatterbox \
  --model mlx-community/chatterbox-fp16 \
  --output .henry/benchmarks/voice/results/chatterbox
```

Pronunciation, prosody, and naturalness require listening evaluation. Automated
results measure model loading, time to first audio, total inference time, and
real-time factor.

## Reports and blind listening

Create `report.md` next to one adapter's JSONL and CSV files:

```bash
uv run python -m tools.voice_benchmark report \
  --results .henry/benchmarks/voice/results/parakeet
```

A parent result directory may contain several adapter directories. The report
command discovers direct result files and results in immediate child
directories:

```bash
uv run python -m tools.voice_benchmark report \
  --results .henry/benchmarks/voice/results
```

Prepare a blind comparison of two or more TTS adapters:

```bash
uv run python -m tools.voice_benchmark tts-review \
  --results \
    .henry/benchmarks/voice/results/piper \
    .henry/benchmarks/voice/results/chatterbox \
  --output .henry/benchmarks/voice/reviews/piper-vs-chatterbox
```

The reviewer completes `ratings.csv` without opening `mapping.json`. Reveal the
model mapping only after listening is complete.

## Recording other people

Use anonymous identifiers such as `speaker-02`. Do not store names, dates of
birth, or contact information in metadata. Voice can be biometric data: obtain
informed consent before recording another person, agree how the recordings will
be stored and deleted, and obtain guardian consent for a child. Never commit or
publish benchmark recordings.
