# Henry

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white)](https://support.apple.com/guide/mac-help/about-this-mac-system-information-mchlp1171/mac)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A local, privacy-first voice assistant built for Apple Silicon Macs.**

Henry keeps wake-word detection, speech recognition, language generation, and speech synthesis on the machine while
logging the live pipeline to the console.

The project is intentionally small and explicit: asyncio coordinates the application, dedicated worker threads own
blocking ML runtimes, the native helper owns full-duplex audio, and ports keep domain services independent from concrete
adapters.

## ✨ Features

- Fully local voice pipeline with no cloud inference.
- Streaming OpenWakeWord detection with Silero VAD.
- Multilingual Parakeet speech recognition, including Polish.
- Conversation history, adaptive response depth, and summarization with LangGraph.
- Direct MLX-LM and LangChain language-model adapters with role-specific models.
- Phrase-streamed Piper speech synthesis.
- Adaptive acoustic and semantic turn endpointing.
- Native macOS voice processing with acoustic echo cancellation, soft ducking,
  and barge-in.
- Console logging with one default local profile.
- Explicit asyncio, worker-thread, port, and adapter boundaries.

## 🎙️ Voice pipeline

```text
Microphone -> Apple Voice Processing (AEC, NS, AGC) -> native PCM channel 0
  -> streaming resampling (16 kHz, 512-sample frames)
  -> Silero VAD + OpenWakeWord
  -> adaptive utterance segmentation
  -> Parakeet STT + semantic continuation detection
  -> response routing + LangGraph + LangChain or MLX-LM
  -> speakable phrase segmentation
  -> Piper TTS (22.05 kHz)
  -> speakers
```

Henry begins in wake-word mode. Activation starts a finite LangGraph run that generates a greeting from the conversation
summary and recent messages. The voice session then remains active for follow-up utterances. Each user turn runs the
reply and summary nodes and stores its state under the in-process `thread_id="default"`.

Short turns use the fast model. Longer or multi-part turns use the detailed
model. The active profile prepares short wake-word and waiting reactions in the
background. A ready wake reaction avoids model latency during activation, while
a waiting reaction is emitted only when a detailed response misses the configured
delay. These reactions are delivery context and are not added to semantic
conversation history.

Model text is divided at natural phrase boundaries so TTS can start before the
whole response exists. During playback, likely user speech first lowers the
assistant volume; sustained speech cancels generation, synthesis, and native
playback. Only phrases confirmed as played are described to the next graph run
as heard by the user.

## 📦 Requirements

- Apple Silicon Mac with Metal support
- macOS default input and output audio devices
- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- Xcode or the Xcode Command Line Tools with Swift

PortAudio, typically installed with `brew install portaudio`, is required only
when using the fallback `pyaudio` audio driver.

The MLX models require real Apple Silicon hardware. Unit tests do not require a microphone, speakers, Metal, ONNX
models, or downloaded Hugging Face models.

## 🛠️ Installation

```bash
uv sync
```

The package build compiles the `henry-audio` Swift helper when its executable
is missing or older than its sources. `./scripts/build-native-audio.sh` can be
used to rebuild it explicitly. The helper owns one full-duplex
`AVAudioEngine` session, so playback is available to Apple's echo canceller as
the reference signal for microphone capture.

`uv sync` installs development tools, including the LangGraph CLI. They are not
part of Henry's runtime dependency set.

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

Profiles use a fixed directory contract:

```text
.henry/profiles/default/
├── profile.yml
└── prompts/
    ├── system.md
    ├── opening.md
    └── summary.md
```

Prompt paths are not configurable. Persona, conversation opening, and summary behavior belong to these profile files.
Versioned Henry and Alexa templates are available in `examples/profiles`. Copy
the selected directories into `.henry/profiles` and keep local model or voice
changes outside version control. Henry is a dry, world-weary but dependable
butler; Alexa is a warm and imaginative assistant intended for children.

## 🚀 Running Henry

Start Henry with the `default` profile and console logging:

```bash
uv run henry-cli
```

The CLI intentionally has no arguments. Use `HENRY_HOME` only to select the local data directory. `Ctrl+C` requests a
clean shutdown.

The default audio driver is configured in `settings.yml`:

```yaml
speech:
  audio:
    driver: avfaudio
  segmentation:
    max_end_silence_frames: 18
    short_utterance_speech_frames: 31
    short_utterance_end_silence_frames: 28
    max_utterance_frames: 1875
```

The conversation adapter is selected independently:

```yaml
conversation:
  adapter: langchain       # or mlx
  acknowledgement_delay: 0.5
  classify_ambiguous: false
```

Each profile maps the `fast`, `detailed`, and optional `classifier` roles to
adapter-specific model identifiers. The supplied templates use the existing
`ollama:gpt-oss:20b` LangChain baseline and
`mlx-community/Qwen3.5-4B-MLX-4bit` for every MLX role on an M1 Max with 64 GB
of unified memory. Roles that use the same model share one loaded instance.
Model weights are not included in the repository. Enable ambiguous-turn
classification after selecting the MLX adapter to classify without adding a
full Ollama round trip. Supplied profiles keep thinking mode disabled so
ordinary spoken replies do not begin with a long silent reasoning phase.
The MLX adapter also caches the stable persona prefix and copies that cache for
each generation; mutable summaries and conversation messages remain outside the
shared cache.

Use `driver: pyaudio` as a fallback when native voice processing is not
available. Developers can point at a separately built helper with
`HENRY_AUDIO_HELPER`; normal installations use the packaged executable.

Alternative speech adapters can be selected in `settings.yml`. They load their
models only when Henry starts:

```yaml
speech:
  stt:
    adapter: mlx:qwen3-asr  # or mlx:parakeet-tdt / mlx:whisper
  tts:
    adapter: mlx:chatterbox # or piper
```

The corresponding `stt.model` and `tts.model` values belong to the active
profile. `stt.language` is an optional adapter-specific hint; set it to `pl`
for Polish Whisper profiles and omit it when Whisper should detect English.
Parakeet and Piper remain the defaults until recorded Polish
benchmarks demonstrate a better choice.

Segmentation values count 512-sample frames at 16 kHz, so one frame is 32 ms.
The defaults use approximately 576 ms of trailing silence for established
speech, 896 ms for a short utterance, and a 60-second hard limit. Tune these
values only after reviewing interaction timing logs and testing real pauses.

## 🏗️ Architecture

| Package              | Responsibility                                                        |
|----------------------|-----------------------------------------------------------------------|
| `henry_speech`       | Audio, wake word, segmentation, STT, TTS, playback, voice session    |
| `henry_conversation` | LangGraph routing, history, summary, model replies, phrase buffering |
| `henry_resources`    | Local profiles, prompts, settings, and model paths                   |
| `henry_cli`          | Default composition root, signals, and console logging               |
| `henry_common`       | Shared lifecycle, events, logging, and validation                     |

The event loop coordinates capture, segmentation, conversation, synthesis,
playback, and control queues. Blocking operations remain with their owners:

- the native audio helper owns AVAudioEngine input, output, voice processing,
  ducking, gain restoration, and immediate playback interruption;
- the speech capture worker owns Silero VAD and OpenWakeWord;
- STT owns Parakeet and its MLX runtime;
- TTS owns Piper;
- conversation owns one language-model executor and keeps model loading,
  generation, and closing in that same thread;
- the selected conversation adapter owns either the local LangChain model or
  direct MLX-LM inference.

Async pipeline boundaries use `asyncio.Queue`. Dedicated ML worker threads
return results through futures and `loop.call_soon_threadsafe(...)`; the native
audio process uses a framed standard-input/standard-output protocol read by one
Python thread.

## 🧪 Development

### Voice benchmarks

Henry includes an interactive recorder, Polish STT/turn-taking/TTS suites,
adapter runners, CSV/JSONL measurements, Markdown reports, and blind TTS
reviews. No model is downloaded while listing prompts or recording audio.

```bash
uv run python -m tools.voice_benchmark list --suite pl-core
uv run python -m tools.voice_benchmark record \
  --suite pl-core --speaker speaker-01 --condition quiet
```

Generated audio and results are kept below `HENRY_HOME/benchmarks` rather than
the repository. The complete recording and evaluation protocol is documented
in [`benchmarks/voice/README.md`](benchmarks/voice/README.md).

### Raw audio loopback

Use the standalone audio diagnostic to hear the raw microphone signal received
by the configured audio driver. It does not initialize VAD, wake word, STT, TTS,
or the conversation pipeline:

```bash
uv run python -m tools.audio_diagnostic --seconds 5
```

The command prints the captured format, RMS level, and peak level before
playing the frame once. To compare the native path with PortAudio, select a
driver explicitly:

```bash
uv run python -m tools.audio_diagnostic --seconds 5 --driver pyaudio
```

To isolate native device capture from Apple's Voice Processing I/O, disable
voice processing for one diagnostic run:

```bash
HENRY_AUDIO_VOICE_PROCESSING=0 \
  uv run python -m tools.audio_diagnostic --seconds 5 --driver avfaudio
```

This switch is diagnostic only. It disables acoustic echo cancellation and
must not be used as the normal Henry configuration.

Measure full-duplex echo cancellation by playing a deterministic test signal
while recording the microphone. Remain silent during the measurement; the
command reports the residual level and plays back what the microphone heard:

```bash
uv run python -m tools.audio_diagnostic \
  --seconds 5 --driver avfaudio --duplex
```

For a meaningful comparison, repeat the same test without Voice Processing and
without changing device selection or volume:

```bash
HENRY_AUDIO_VOICE_PROCESSING=0 \
  uv run python -m tools.audio_diagnostic \
  --seconds 5 --driver avfaudio --duplex
```

`relative_to_playback` is not a calibrated acoustic ERLE measurement. Use the
difference between the two runs to evaluate echo suppression on one device
setup.

### Automated verification

Conversation model latency and routing can be measured without audio:

```bash
uv run python -m tools.conversation_benchmark --adapter langchain
uv run python -m tools.conversation_benchmark --adapter mlx
```

See `benchmarks/conversation/README.md` for the Polish source suite, report
format, and human listening criteria. Generated results are local artifacts and
must not be committed.

Run the complete automated verification:

```bash
uv run ruff check hatch_build.py src tests tools
uv run ruff format --check hatch_build.py src tests tools pyproject.toml
uv run pytest -q
uv run python -m compileall -q hatch_build.py src tests tools
swift format lint --recursive native/macos/henry-audio
swift test --package-path native/macos/henry-audio
uv build
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
- AVFAudio uses the current default macOS input and output devices.
- Native helper startup and acoustic behavior still require a manual
  microphone/speaker smoke test on the target Mac.
- Acoustic endpoint thresholds are frame-based. Semantic continuation detection
  is deliberately conservative and does not start speculative LLM work.
- Playback delivery tracking confirms complete spoken phrases, not the exact
  sample or word at which an interrupted phrase stopped.

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
