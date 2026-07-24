# Henry

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white)](https://support.apple.com/guide/mac-help/about-this-mac-system-information-mchlp1171/mac)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A local, privacy-first voice assistant built for Apple Silicon Macs.**

Henry keeps wake-word detection, speech recognition, language generation, and
speech synthesis on the machine while exposing the live pipeline through either
a Textual terminal UI or console event logging.

The project is intentionally small and explicit: asyncio coordinates the
application, dedicated worker threads own blocking audio and ML runtimes, and
ports keep domain services independent from concrete adapters.

## ✨ Features

- Fully local voice pipeline with no cloud inference.
- Streaming OpenWakeWord detection with Silero VAD.
- Multilingual Parakeet speech recognition, including Polish.
- MLX language-model inference optimized for Apple Silicon.
- Line-buffered Piper speech synthesis.
- Textual terminal UI with an optional lightweight event-logging mode.
- Explicit asyncio, worker-thread, port, and adapter boundaries.

## 🎙️ Voice pipeline

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
conversation session active for follow-ups and returns to wake-word mode after a
configurable number of consecutive empty utterance timeouts.

## 📦 Requirements

- Apple Silicon Mac with Metal support
- macOS default input and output audio devices
- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- PortAudio, typically installed with `brew install portaudio`

The MLX models require real Apple Silicon hardware. Unit tests do not require a
microphone, speakers, Metal, ONNX models, or downloaded Hugging Face models.

## 🛠️ Installation

```bash
uv sync
```

## 🧠 Model setup

Download the MLX and Piper models before starting Henry. The `hf` CLI is provided
by the installed `huggingface-hub` dependency, and `uv run hf download` stores
models in the same Hugging Face cache that Henry's adapters use at runtime.

### 🗂️ Model inventory

| Pipeline stage | Model or repository                                                                                    | Purpose and recommendation                                                         |
|----------------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Wake word      | Custom `.onnx` from [openwakeword.com](https://openwakeword.com/)                                      | Recommended source for a custom activation phrase                                  |
| Wake word      | [`alexa_v0.1.onnx`](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.onnx) | Official pre-trained openWakeWord model for "Alexa"; useful for testing            |
| VAD            | [`mlx-community/silero-vad`](https://huggingface.co/mlx-community/silero-vad)                          | Fixed model used for voice activity detection                                      |
| STT            | [`mlx-community/parakeet-tdt-0.6b-v3`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3)      | Fixed multilingual speech-to-text model, including Polish                          |
| Language model | [`mlx-community/Qwen3.5-4B-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit)        | Smaller and lower-quality model; suitable for diagnostics                          |
| Language model | [`mlx-community/Qwen3.5-9B-OptiQ-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-OptiQ-4bit)    | Recommended default with sufficient response quality                               |
| Voice          | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices/tree/main)                        | Repository containing Piper voices; Henry supports repository-relative model paths |

### 🤗 Downloading Hugging Face models

Download the VAD, STT, and the language model you want to run:

```bash
uv run hf download mlx-community/silero-vad
uv run hf download mlx-community/parakeet-tdt-0.6b-v3
uv run hf download mlx-community/Qwen3.5-4B-MLX-4bit
uv run hf download mlx-community/Qwen3.5-9B-OptiQ-4bit
```

Piper voices consist of an ONNX file and its adjacent `.onnx.json`
configuration. High quality is recommended; medium quality is a smaller
alternative.

| Quality            | `--voice-model` / `HENRY_VOICE_MODEL` value     | Download command                                                                                                                           |
|--------------------|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Medium             | `pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx` | `uv run hf download rhasspy/piper-voices pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json` |
| High (recommended) | `pl/pl_PL/bass/high/pl_PL-bass-high.onnx`       | `uv run hf download rhasspy/piper-voices pl/pl_PL/bass/high/pl_PL-bass-high.onnx pl/pl_PL/bass/high/pl_PL-bass-high.onnx.json`             |

### 👂 Downloading OpenWakeWord models

OpenWakeWord models do not use the Hugging Face cache. Henry expects the wake-word
model and the two shared feature models in its data directory:

```text
.henry/
└── models/
    └── openwakeword/
        ├── embedding_model.onnx
        ├── melspectrogram.onnx
        └── <wake-word-model>.onnx
```

The official openWakeWord project provides pre-trained models through its
[GitHub releases](https://github.com/dscripka/openWakeWord/releases). For
example, install the pre-trained Alexa model and its required feature models with:

```bash
mkdir -p .henry/models/openwakeword
curl -L https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx \
  -o .henry/models/openwakeword/melspectrogram.onnx
curl -L https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx \
  -o .henry/models/openwakeword/embedding_model.onnx
curl -L https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.onnx \
  -o .henry/models/openwakeword/alexa_v0.1.onnx
```

Models created at [openwakeword.com](https://openwakeword.com/) should be copied
to the same directory and selected by filename with `--wakeword-model` or
`HENRY_WAKEWORD_MODEL`.

The data directory is resolved in this order:

1. `HENRY_HOME`, when set.
2. The nearest `.henry` directory in the current directory or one of its parents.
3. The platform-specific user data directory returned by `platformdirs`.

Local `.henry` data and downloaded models are not part of the Python package.

## 🚀 Running Henry

Start the terminal UI:

```bash
uv run henry-cli
```

Run without the terminal UI and log application events to the console:

```bash
uv run henry-cli -noui
```

The conventional long form `--no-ui` is also supported. Both modes handle
`SIGINT` and `SIGTERM`. Press `q` in the terminal UI or use `Ctrl+C` to request
shutdown.

Profiles configure the assistant name, system prompt style, wake-word model,
spoken activation reply, and Piper voice. Command-line arguments take precedence
over environment variables, which take precedence over the application defaults.

| Command-line argument  | Environment variable        | Default                                   |
|------------------------|-----------------------------|-------------------------------------------|
| `-noui` / `--no-ui`    | —                           | disabled                                  |
| `--log-level`          | `HENRY_LOG_LEVEL`           | `DEBUG`                                   |
| `--profile-kind`       | `HENRY_PROFILE_KIND`        | `default`                                 |
| `--profile-name`       | `HENRY_PROFILE_NAME`        | `Henry`                                   |
| `--system-language`    | `HENRY_SYSTEM_LANGUAGE`     | `Polish`                                  |
| `--wakeword-reply`     | `HENRY_WAKEWORD_REPLY`      | `Tak, Wielmożny Panie...`                 |
| `--wakeword-model`     | `HENRY_WAKEWORD_MODEL`      | `Hey_Henree_20260406_162745.onnx`         |
| `--voice-model`        | `HENRY_VOICE_MODEL`         | `pl/pl_PL/bass/high/pl_PL-bass-high.onnx` |
| `--language-model`     | `HENRY_LANGUAGE_MODEL`      | `mlx-community/Qwen3.5-9B-OptiQ-4bit`     |
| `--max-empty-segments` | `HENRY_MAX_EMPTY_SEGMENTS`  | `3`                                       |

For example:

```bash
HENRY_PROFILE_NAME=Ada uv run henry-cli --language-model local/model
uv run henry-cli -noui --log-level TRACE --wakeword-model alexa_v0.1.onnx
```

Run `uv run henry-cli --help` for the complete argument reference.

## 🏗️ Architecture

| Package           | Responsibility                                                     |
|-------------------|--------------------------------------------------------------------|
| `henry_client`    | Domain models, ports, adapters, services, and orchestration        |
| `henry_cli`       | Textual UI, console diagnostics, telemetry state, and buffered logs |
| `henry_resources` | Resolution of local data and model paths                           |

The event loop runs capture, transcription, processing, and replay tasks. Blocking
operations live in long-running workers:

- audio input owns PyAudio input, Silero VAD, and OpenWakeWord;
- audio output owns the PyAudio output stream;
- STT owns Parakeet and its MLX runtime;
- TTS owns Piper;
- reply owns the MLX language model.

Worker-side requests use `queue.Queue`. Results return to asyncio queues or
futures through `loop.call_soon_threadsafe(...)`.

## 🧪 Development

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

`pytest` enforces at least 95% branch coverage across the CLI, client, and
resource packages. Concrete adapter modules are excluded from that threshold and
remain subject to focused contract tests and manual Apple Silicon hardware checks.

For environments where the default uv cache is read-only, prefix commands with:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache
```

## ⚠️ Current limitations

- macOS and Apple Silicon are the only supported runtime target.
- Input and output use the current default PortAudio devices.
- The application has no acoustic echo cancellation or barge-in support.
- A conversation session currently has no inactivity timeout.
- Utterance segmentation currently relies on trailing silence and has no hard
  maximum recording duration.

Hardware behavior still requires a manual smoke test. Passing fake-stream tests
does not prove that a particular microphone, output device, or aggregate audio
configuration is compatible.

## 📄 License

Henry's source code is available under the [MIT License](LICENSE).

Dependencies and models are separate works and remain subject to their own
licenses. They are downloaded separately and are not relicensed under Henry's
MIT License.

| Component                                                                  | License                                                                                                                     |
|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [openWakeWord](https://github.com/dscripka/openWakeWord) code              | Apache 2.0                                                                                                                  |
| Official pre-trained openWakeWord models, including `alexa_v0.1.onnx`      | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — non-commercial use and ShareAlike requirements apply |
| [Piper voices](https://huggingface.co/rhasspy/piper-voices)                 | MIT                                                                                                                         |
| [Parakeet TDT 0.6B v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) | CC BY 4.0 — attribution required                                                                                     |
| [Qwen 3.5 4B MLX](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) | Apache 2.0                                                                                                                  |
| [Qwen 3.5 9B OptiQ](https://huggingface.co/mlx-community/Qwen3.5-9B-OptiQ-4bit) | Apache 2.0                                                                                                               |

Custom wake-word models and alternative model variants may use different
licenses. Check the relevant model card or download source before redistribution
or commercial use.
