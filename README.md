# Henry

Henry is a local, privacy-first voice assistant for Apple Silicon Macs. It keeps
speech detection, transcription, language generation, and speech synthesis on
the machine while exposing the live pipeline through a Textual terminal UI.

The project is intentionally small and explicit: asyncio coordinates the
application, dedicated worker threads own blocking audio and ML runtimes, and
ports keep domain services independent from concrete adapters.

## Voice pipeline

```text
Microphone (16 kHz)
  -> Silero VAD + OpenWakeWord
  -> utterance segmentation
  -> Parakeet STT
  -> Qwen / MLX language model
  -> line-buffered reply
  -> Piper TTS (22.05 kHz)
  -> speakers
```

Henry begins in wake-word mode. After activation it plays a preloaded spoken
acknowledgement and enters utterance mode. The current implementation keeps the
conversation session active after that first activation; it does not require the
wake word again before every follow-up.

## Requirements

- Apple Silicon Mac with Metal support
- macOS default input and output audio devices
- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- PortAudio, typically installed with `brew install portaudio`

The MLX models require real Apple Silicon hardware. Unit tests do not require a
microphone, speakers, Metal, ONNX models, or downloaded Hugging Face models.

## Installation

```bash
uv sync
```

Piper voices and MLX models are downloaded and cached by their respective
libraries on first use. OpenWakeWord feature and wake-word models are expected in
Henry's data directory:

```text
.henry/
└── models/
    └── openwakeword/
        ├── embedding_model.onnx
        ├── melspectrogram.onnx
        └── <wake-word-model>.onnx
```

The data directory is resolved in this order:

1. `HENRY_HOME`, when set.
2. The nearest `.henry` directory in the current directory or one of its parents.
3. The platform-specific user data directory returned by `platformdirs`.

Local `.henry` data and downloaded models are not part of the Python package.

## Running Henry

Start the terminal UI:

```bash
uv run henry-cli
```

Start the event-logging debugger:

```bash
uv run henry-debugger
```

Both applications handle `SIGINT` and `SIGTERM`. Press `q` in the terminal UI or
use `Ctrl+C` to request shutdown.

Profiles configure the assistant name, system prompt style, wake-word model,
spoken activation reply, and Piper voice. The current CLI and debugger profiles
live in `src/henry_cli/main.py` and `src/henry_debugger/main.py`.

## Architecture

| Package | Responsibility |
| --- | --- |
| `henry_client` | Domain models, ports, adapters, services, and orchestration |
| `henry_cli` | Textual UI, telemetry snapshots, pipeline state, and buffered logs |
| `henry_debugger` | Lightweight event and lifecycle diagnostics |
| `henry_common` | Event-loop/thread handoff helpers |
| `henry_resources` | Resolution of local data and model paths |

The event loop runs capture, transcription, processing, and replay tasks. Blocking
operations live in long-running workers:

- audio input owns PyAudio input, Silero VAD, and OpenWakeWord;
- audio output owns the PyAudio output stream;
- STT owns Parakeet and its MLX runtime;
- TTS owns Piper;
- conversation owns the MLX language model.

Worker-side requests use `queue.Queue`. Results return to asyncio queues or
futures through `loop.call_soon_threadsafe(...)`.

## Development

Run the complete automated verification:

```bash
uv run ruff check src tests
uv run ruff format --check src tests pyproject.toml
uv run pytest -q
uv run python -m compileall -q src tests
```

The tests progress from pure domain behavior to service lifecycle and complete
orchestration. Fake ports exercise real queues, threads, and asyncio tasks without
opening hardware or loading production models.

For environments where the default uv cache is read-only, prefix commands with:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache
```

## Current limitations

- macOS and Apple Silicon are the only supported runtime target.
- Input and output use the current default PortAudio devices.
- The application has no acoustic echo cancellation or barge-in support.
- A conversation session currently has no inactivity timeout.
- Utterance segmentation currently relies on trailing silence and has no hard
  maximum recording duration.

Hardware behavior still requires a manual smoke test. Passing fake-stream tests
does not prove that a particular microphone, output device, or aggregate audio
configuration is compatible.
