# Henry

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white)](https://support.apple.com/guide/mac-help/about-this-mac-system-information-mchlp1171/mac)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A local-first voice assistant built for Apple Silicon Macs.**

Henry runs wake-word detection, speech recognition, language generation, and speech synthesis on the Mac by default. A
terminal interface shows the live conversation, runtime state, audio signals, model-loading progress, latency, and logs
without moving voice-session decisions out of the speech domain.

The project favors explicit ownership over framework-heavy orchestration:
asyncio coordinates the application, dedicated worker threads own blocking ML runtimes, and a native Swift helper owns
the full-duplex macOS audio session.

## What Henry does

- Detects a custom wake word with OpenWakeWord and Silero VAD.
- Keeps a voice session active for follow-up turns after activation.
- Transcribes multilingual speech locally with Parakeet by default.
- Routes short and detailed responses through finite LangGraph runs.
- Preserves process-lifetime conversation history and summaries.
- Streams complete phrases into Piper speech synthesis.
- Uses Apple Voice Processing for echo cancellation, ducking, and barge-in.
- Displays conversation progress, model state, devices, signals, timings, and logs in a Textual interface.
- Supports alternative STT, TTS, audio, and conversation-model adapters without loading unused runtimes.

## Requirements

- Apple Silicon Mac with Metal support
- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- Xcode or the Xcode Command Line Tools with Swift
- macOS input and output audio devices

PortAudio, commonly installed with `brew install portaudio`, is required only for the fallback `pyaudio` driver.

## Quick start

Initialize a checkout:

```bash
make init
```

This command:

1. builds the native `henry-audio` helper when required;
2. synchronizes the Python environment;
3. creates `.henry` and installs the versioned settings, default profile, and custom wake-word model without replacing
   existing files;
4. downloads the models selected by that configuration.

Re-running initialization is safe. Existing local files are preserved and the Hugging Face cache is reused.

Start Henry:

```bash
make run
```

The terminal interface first validates the available profiles and asks which one to use. Audio devices and models are
opened only after a valid profile is selected. During startup, model downloads and loading progress are displayed in the
interface; a failed startup can be retried without restarting the program.

The main shortcuts are:

| Key | Action                                   |
|-----|------------------------------------------|
| `c` | Show the conversation                    |
| `l` | Show runtime logs                        |
| `i` | Toggle runtime information and telemetry |
| `q` | Quit                                     |

The application intentionally has no runtime arguments. Set `HENRY_HOME` when you need to select a different local data
directory. `Ctrl+C` and `q` request a clean shutdown.

## Voice flow

```text
Microphone
  -> Apple Voice Processing (AEC, noise suppression, gain control)
  -> streaming 16 kHz mono capture
  -> Silero VAD + OpenWakeWord
  -> adaptive utterance segmentation
  -> Parakeet STT + semantic continuation detection
  -> response routing + finite LangGraph run
  -> MLX-LM or LangChain language model
  -> speakable phrase segmentation
  -> Piper TTS
  -> native playback
```

One wake-word detection activates the voice session. Activation generates an opening, while subsequent utterances
generate replies without requiring the wake word again. Conversation runs are finite; the long-running workers wait for
future speech.

Short turns use the fast model and longer or multi-part turns use the detailed model. The selected profile may provide
prepared wake and waiting reactions. These are delivery-only phrases and do not become semantic conversation history.

Generated text is divided at natural phrase boundaries so synthesis can begin before the complete response exists.
Likely user speech first lowers playback gain; sustained speech interrupts generation, synthesis, and playback. Only
phrases confirmed as played are reported to the next conversation run as heard by the user.

## Configuration and profiles

Runtime configuration lives below the local Henry data directory:

```text
.henry/
├── settings.yml
├── models/
│   └── openwakeword/
└── profiles/
    └── <profile-id>/
        ├── profile.yml
        ├── prompts/
        │   ├── system.md
        │   ├── opening.md
        │   └── summary.md
        └── reactions/
            ├── wake.txt
            └── wait.txt
```

`settings.yml` selects technical adapters and their runtime tuning. A profile owns the assistant identity, prompts,
reactions, voice, wake word, and model parameters supported by the selected adapters. Prompt and reaction paths are a
fixed contract rather than configurable YAML fields.

The data root is resolved from `HENRY_HOME`, the nearest `.henry` directory, or the macOS user data directory, in that
order. Local data, model weights, recordings, and generated benchmark results are not committed.

See [Configuration](docs/configuration.md) for the complete settings/profile contract, model inventory, adapter
selection, and manual model setup.

## Architecture

| Package              | Responsibility                                                                                                 |
|----------------------|----------------------------------------------------------------------------------------------------------------|
| `henry_speech`       | Audio, VAD, wake word, utterance segmentation, STT, TTS, playback, and voice-session transitions               |
| `henry_conversation` | Conversation inputs, LangGraph routing and state, history, summaries, model generation, and reply segmentation |
| `henry_resources`    | Discovery and loading of local settings, profiles, prompts, and model paths                                    |
| `henry_cli`          | Composition root, profile selection, terminal UI, logging, signals, startup, and shutdown                      |
| `henry_common`       | Shared lifecycle, events, logging, and validation primitives                                                   |

The asyncio event loop coordinates domain workers through explicit events and queues. Blocking audio and ML operations
remain in their owning processes or threads. The terminal UI is a projection of domain events and telemetry; it does not
control conversation or speech decisions.

The native [`henry-audio`](native/macos/henry-audio/README.md) helper owns one full-duplex `AVAudioEngine` process for
capture, playback, Apple Voice Processing, gain control, and immediate interruption.

Repository design and implementation conventions are documented in
[Engineering Guidelines](ENGINEERING.md).

## Development

Run the complete automated verification:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff check hatch_build.py src tests tools
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff format --check hatch_build.py src tests tools pyproject.toml
UV_CACHE_DIR=/private/tmp/uv-cache uv run pyrefly check
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run python -m compileall -q hatch_build.py src tests tools
swift format lint --recursive native/macos/henry-audio
swift test --package-path native/macos/henry-audio
UV_CACHE_DIR=/private/tmp/uv-cache uv build
```

The Python suite uses fake ports, real queues, threads, and asyncio tasks. It does not require microphones, speakers,
Metal, production model files, model downloads, Ollama, or real PyAudio devices. `pytest` enforces at least 95% combined
branch coverage across the five production packages; concrete adapter modules are excluded from that threshold.

Additional tools:

```bash
# Inspect the configured audio path without starting the assistant.
uv run python -m tools.audio_diagnostic --seconds 5

# Run the selected conversation model against the Polish core suite.
uv run python -m tools.conversation_benchmark

# List voice-benchmark prompts without loading models or opening audio.
uv run python -m tools.voice_benchmark list --suite pl-core
```

See the [voice benchmark protocol](benchmarks/voice/README.md) and
[conversation benchmark protocol](benchmarks/conversation/README.md) for recording, evaluation, reports, and acceptance
boundaries. Generated artifacts belong below `HENRY_HOME/benchmarks` and must not be committed.

## Current limitations

- macOS on Apple Silicon is the only supported runtime target.
- AVFAudio uses the current default macOS input and output devices.
- Conversation memory is in-process and is lost when Henry exits.
- Acoustic endpoint thresholds remain frame-based; semantic continuation is conservative and does not start speculative
  model work.
- Delivery tracking confirms complete phrases, not the exact word or sample at which an interrupted phrase stopped.
- Native helper startup, microphones, speakers, echo cancellation, gain ramps, Metal, and real models still require
  manual validation on the target Mac.

Passing automated tests does not prove that a particular audio device setup or model behaves correctly in a real voice
session.

## License

Henry's source code is available under the [MIT License](LICENSE).

Dependencies and models are separate works and retain their own licenses. In particular, voices, wake-word assets, and
individual Hugging Face models may impose attribution, redistribution, or usage conditions. Review the relevant model
card or download source before redistribution or commercial use.
