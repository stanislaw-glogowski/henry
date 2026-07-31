# Henry

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white)](https://support.apple.com/guide/mac-help/about-this-mac-system-information-mchlp1171/mac)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A local, privacy-first voice assistant built for Apple Silicon Macs.**

Henry keeps wake-word detection, speech recognition, language generation, and speech synthesis on the machine while
logging the live pipeline to the console.

The project is intentionally small and explicit: asyncio coordinates the application, dedicated worker threads own
blocking audio and ML runtimes, and ports keep domain services independent from concrete adapters.

## ✨ Features

- Fully local voice pipeline with no cloud inference.
- Streaming OpenWakeWord detection with Silero VAD.
- Multilingual Parakeet speech recognition, including Polish.
- Conversation history and summarization with LangGraph and local Ollama.
- Line-buffered Piper speech synthesis.
- Console logging with one default local profile.
- Explicit asyncio, worker-thread, port, and adapter boundaries.

## 🎙️ Voice pipeline

```text
Microphone (16 kHz)
  -> Silero VAD + OpenWakeWord
  -> utterance segmentation
  -> Parakeet STT
  -> LangGraph + Ollama
  -> line-buffered reply
  -> Piper TTS (22.05 kHz)
  -> speakers
```

Henry begins in wake-word mode. Activation starts a finite LangGraph run that generates a greeting from the conversation
summary and recent messages. The voice session then remains active for follow-up utterances. Each user turn runs the
reply and summary nodes and stores its state under the in-process `thread_id="default"`.

## 📦 Requirements

- Apple Silicon Mac with Metal support
- macOS default input and output audio devices
- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- PortAudio, typically installed with `brew install portaudio`

The MLX models require real Apple Silicon hardware. Unit tests do not require a microphone, speakers, Metal, ONNX
models, or downloaded Hugging Face models.

## 🛠️ Installation

```bash
uv sync
```

## 🧠 Model setup

Download the speech and Piper models and make the configured Ollama model available before starting Henry. The `hf` CLI
is provided by the installed `huggingface-hub` dependency, and `uv run hf download` stores speech models in the same
Hugging Face cache that Henry's adapters use at runtime.

### 🗂️ Model inventory

| Pipeline stage | Model or repository                                                                                    | Purpose and recommendation                                                         |
|----------------|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Wake word      | Custom `.onnx` from [openwakeword.com](https://openwakeword.com/)                                      | Recommended source for a custom activation phrase                                  |
| Wake word      | [`alexa_v0.1.onnx`](https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.onnx) | Official pre-trained openWakeWord model for "Alexa"; useful for testing            |
| VAD            | [`mlx-community/silero-vad`](https://huggingface.co/mlx-community/silero-vad)                          | Fixed model used for voice activity detection                                      |
| STT            | [`mlx-community/parakeet-tdt-0.6b-v3`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3)      | Fixed multilingual speech-to-text model, including Polish                          |
| Language model | `ollama:gpt-oss:20b`                                                                                 | Default local conversation model                                                    |
| Voice          | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices/tree/main)                        | Repository containing Piper voices; Henry supports repository-relative model paths |

### 🤗 Downloading Hugging Face models

Download the VAD and STT models:

```bash
uv run hf download mlx-community/silero-vad
uv run hf download mlx-community/parakeet-tdt-0.6b-v3
```

Piper voices consist of an ONNX file and its adjacent `.onnx.json`
configuration. High quality is recommended; medium quality is a smaller alternative.

| Quality            | `tts.model` value in `profile.yml`               | Download command                                                                                                                           |
|--------------------|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Medium             | `pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx` | `uv run hf download rhasspy/piper-voices pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json` |
| High (recommended) | `pl/pl_PL/bass/high/pl_PL-bass-high.onnx`       | `uv run hf download rhasspy/piper-voices pl/pl_PL/bass/high/pl_PL-bass-high.onnx pl/pl_PL/bass/high/pl_PL-bass-high.onnx.json`             |

### 👂 Downloading OpenWakeWord models

OpenWakeWord models do not use the Hugging Face cache. Henry expects the wake-word model and the two shared feature
models in its data directory:

```text
.henry/
└── models/
    └── openwakeword/
        ├── embedding_model.onnx
        ├── melspectrogram.onnx
        └── <wake-word-model>.onnx
```

The official openWakeWord project provides pre-trained models through its
[GitHub releases](https://github.com/dscripka/openWakeWord/releases). For example, install the pre-trained Alexa model
and its required feature models with:

```bash
mkdir -p .henry/models/openwakeword
curl -L https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx \
  -o .henry/models/openwakeword/melspectrogram.onnx
curl -L https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx \
  -o .henry/models/openwakeword/embedding_model.onnx
curl -L https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.onnx \
  -o .henry/models/openwakeword/alexa_v0.1.onnx
```

Models created at [openwakeword.com](https://openwakeword.com/) should be copied to the same directory and selected in
the default profile.

The data directory is resolved in this order:

1. `HENRY_HOME`, when set.
2. The nearest `.henry` directory in the current directory or one of its parents.
3. The platform-specific user data directory returned by `platformdirs`.

Local `.henry` data and downloaded models are not part of the Python package.

The default profile uses a fixed directory contract:

```text
.henry/profiles/default/
├── profile.yml
└── prompts/
    ├── system.md
    ├── opening.md
    └── summary.md
```

Prompt paths are not configurable. Persona, conversation opening, and summary behavior belong to these profile files.

## 🚀 Running Henry

Start Henry with the `default` profile and console logging:

```bash
uv run henry-cli
```

The CLI intentionally has no arguments. Use `HENRY_HOME` only to select the local data directory. `Ctrl+C` requests a
clean shutdown.

## 🏗️ Architecture

| Package              | Responsibility                                                        |
|----------------------|-----------------------------------------------------------------------|
| `henry_speech`       | Audio, wake word, segmentation, STT, TTS, playback, voice session    |
| `henry_conversation` | LangGraph routing, history, summary, model replies, line buffering   |
| `henry_resources`    | Local profiles, prompts, settings, and model paths                   |
| `henry_cli`          | Default composition root, signals, and console logging               |
| `henry_common`       | Shared lifecycle, events, logging, and validation                     |

The event loop runs capture, transcription, processing, and replay tasks. Blocking operations live in long-running
workers:

- audio input owns PyAudio input, Silero VAD, and OpenWakeWord;
- audio output owns the PyAudio output stream;
- STT owns Parakeet and its MLX runtime;
- TTS owns Piper;
- conversation calls the local Ollama model through LangChain.

Worker-side requests use `queue.Queue`. Results return to asyncio queues or futures through
`loop.call_soon_threadsafe(...)`.

## 🧪 Development

Run the complete automated verification:

```bash
uv run ruff check src tests
uv run ruff format --check src tests pyproject.toml
uv run pytest -q
uv run python -m compileall -q src tests
```

The tests progress from pure domain behavior to service lifecycle and complete orchestration. Fake ports exercise real
queues, threads, and asyncio tasks without opening hardware or loading production models.

`pytest` enforces at least 95% combined coverage with branch measurement across all source packages.
Concrete adapter modules are excluded from that threshold and remain subject to focused contract tests and manual Apple
Silicon hardware checks. The suite uses fake-driven graph and service tests with real asyncio queues, tasks, and worker
threads.

For environments where the default uv cache is read-only, prefix commands with:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache
```

## ⚠️ Current limitations

- macOS and Apple Silicon are the only supported runtime target.
- Input and output use the current default PortAudio devices.
- The application has no acoustic echo cancellation or barge-in support.
- Session expiry is based on consecutive empty utterance windows, not wall-clock inactivity.
- Utterance segmentation currently relies on trailing silence and has no hard maximum recording duration.

Hardware behavior still requires a manual smoke test. Passing fake-stream tests does not prove that a particular
microphone, output device, or aggregate audio configuration is compatible.

## 📄 License

Henry's source code is available under the [MIT License](LICENSE).

Dependencies and models are separate works and remain subject to their own licenses. They are downloaded separately and
are not relicensed under Henry's MIT License.

| Component                                                                         | License                                                                                                                      |
|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [openWakeWord](https://github.com/dscripka/openWakeWord) code                     | Apache 2.0                                                                                                                   |
| Official pre-trained openWakeWord models, including `alexa_v0.1.onnx`             | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — non-commercial use and ShareAlike requirements apply |
| [Piper voices](https://huggingface.co/rhasspy/piper-voices)                       | MIT                                                                                                                          |
| [Parakeet TDT 0.6B v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) | CC BY 4.0 — attribution required                                                                                             |

Custom wake-word models and alternative model variants may use different licenses. Check the relevant model card or
download source before redistribution or commercial use.
