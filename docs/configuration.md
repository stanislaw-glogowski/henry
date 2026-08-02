# Configuration

Henry separates technical runtime settings from profile-specific behavior. This keeps adapter selection independent from
assistant identity and prevents unused provider configuration from leaking into the active runtime.

## Local data directory

Henry resolves its data root in this order:

1. `HENRY_HOME`, when set;
2. the nearest `.henry` directory in the current directory or one of its parents;
3. the macOS user data directory returned by `platformdirs`.

The repository initializer uses the checkout-local `.henry` directory unless
`HENRY_HOME` is set:

```bash
make init
```

Initialization copies versioned defaults without replacing existing files and downloads the models required by the
installed `settings.yml` and `default`
profile.

## Directory contract

```text
<henry-data>/
├── settings.yml
├── models/
│   └── openwakeword/
│       ├── embedding_model.onnx
│       ├── melspectrogram.onnx
│       ├── silero_vad.onnx
│       └── <wake-word-model>.onnx
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

Prompt and reaction paths are fixed. Reaction files contain one phrase per non-empty line. Additional profiles may be
added as sibling directories below
`profiles/`; the terminal interface validates and lists them at startup.

The versioned templates are available in [`examples`](../examples). Keep local persona edits, model choices, downloaded
weights, and recordings outside version control.

## Settings and profiles

`settings.yml` contains technical selections and tuning shared by every profile. The profile contains only parameters
interpreted by the selected adapters, together with persona and delivery content.

The default technical settings are equivalent to:

```yaml
conversation:
  language_model:
    adapter: mlx
  acknowledgement_delay: 0.5
  classify_ambiguous: false

speech:
  audio:
    driver: avfaudio
  vad:
    adapter: mlx:silero_vad
    threshold: 0.5
  wakeword:
    adapter: openwakeword
  stt:
    adapter: mlx:parakeet-tdt
    model_id: mlx-community/parakeet-tdt-0.6b-v3
  tts:
    adapter: piper
    repo_id: rhasspy/piper-voices
    normalize_audio: true
    volume: 1.0
```

The supplied profile selects the custom wake word, a Polish Piper voice, and the model roles used by the conversation
adapter:

```yaml
name: Henry

conversation:
  models:
    fast:
      model_id: mlx-community/Qwen3.5-4B-MLX-4bit
      max_tokens: 128
      temperature: 0.5
    detailed:
      model_id: mlx-community/Qwen3.5-4B-MLX-4bit
      max_tokens: 256
      temperature: 0.6
    classifier:
      model_id: mlx-community/Qwen3.5-4B-MLX-4bit
      max_tokens: 4
      temperature: 0.0
  recent_messages: 8

wakeword:
  label: Hey Henry
  model_path: hey_henry.onnx

tts:
  model_path: pl/pl_PL/bass/high/pl_PL-bass-high.onnx
```

The classifier role is loaded only when `classify_ambiguous` is enabled. Roles that use the same MLX identifier share
one loaded model instance.

## Conversation adapters

Direct local MLX-LM inference is the default:

```yaml
conversation:
  language_model:
    adapter: mlx
```

MLX model roles accept `model_id`, `max_tokens`, `temperature`, `top_p`,
`top_k`, and `thinking`. Supplied profiles keep thinking disabled so spoken responses do not begin with a long reasoning
delay.

To use a LangChain chat model exposed through an Ollama-compatible endpoint:

```yaml
conversation:
  language_model:
    adapter: langchain
    base_url: http://localhost:11434
```

Replace the active profile's model identifiers with identifiers understood by that provider, for example
`ollama:gpt-oss:20b`. LangChain roles do not accept the MLX-specific `top_k` field. Keep only parameters supported by
the selected adapter in a profile.

The default configuration is local. A custom `base_url` can point elsewhere, so review that endpoint before treating an
alternative setup as private or offline.

## Speech adapters

The native full-duplex driver is the default:

```yaml
speech:
  audio:
    driver: avfaudio
```

Use `driver: pyaudio` as a fallback when native Apple Voice Processing is not available. PyAudio requires PortAudio and
does not provide Henry's native speaker-reference echo cancellation. Developers can point at a separately built native
executable with `HENRY_AUDIO_HELPER`; normal installations use the packaged helper.

Available speech-model adapters are selected in `settings.yml`:

```yaml
speech:
  stt:
    adapter: mlx:qwen3-asr
    model_id: mlx-community/Qwen3-ASR-0.6B-8bit
  tts:
    adapter: mlx:chatterbox
    model_id: mlx-community/chatterbox-fp16
    lang_code: en
```

STT adapters are `mlx:parakeet-tdt`, `mlx:qwen3-asr`, and `mlx:whisper`. TTS adapters are `piper` and `mlx:chatterbox`.
Each settings variant owns its technical defaults. A profile may override only fields supported by the selected adapter:

- Piper uses `tts.model_path` and may override its repository and voice tuning.
- Chatterbox may override `tts.model_id` and `tts.lang_code`.
- Each STT profile may override `stt.model_id`.
- Whisper alone accepts the optional `stt.language` hint.

Parakeet and Piper remain the supplied defaults until recorded Polish benchmarks demonstrate a better choice and, for
TTS, a blind listening review confirms it.

## Segmentation

The exposed capture stream contains 512-sample mono frames at 16 kHz, so one frame represents 32 milliseconds. The
supplied segmentation settings are:

```yaml
speech:
  segmentation:
    min_start_speech_frames: 10
    max_start_silence_frames: 150
    max_end_silence_frames: 18
    short_utterance_speech_frames: 31
    short_utterance_end_silence_frames: 28
    max_utterance_frames: 1875
    pre_roll_frames: 15
```

These values provide approximately 576 ms of trailing silence for established speech, 896 ms for short utterances, and a
60-second hard limit. Tune them only after reviewing interaction timing and real speech pauses; they affect the
conversation feel directly.

## Model inventory

| Stage        | Supplied model or repository                                                   |
|--------------|--------------------------------------------------------------------------------|
| Wake word    | `examples/models/openwakeword/hey_henry.onnx` plus OpenWakeWord feature models |
| VAD          | `mlx-community/silero-vad`                                                     |
| STT          | `mlx-community/parakeet-tdt-0.6b-v3`                                           |
| Conversation | `mlx-community/Qwen3.5-4B-MLX-4bit`                                            |
| TTS          | `rhasspy/piper-voices` with `pl_PL-bass-high`                                  |

`make init` installs the custom wake-word asset, downloads the shared OpenWakeWord ONNX files, resolves complete Hugging
Face repositories, and downloads both the selected Piper `.onnx` file and its adjacent `.onnx.json`
configuration.

Equivalent manual Hugging Face commands for the supplied defaults include:

```bash
uv run hf download mlx-community/Qwen3.5-4B-MLX-4bit
uv run hf download mlx-community/silero-vad
uv run hf download mlx-community/parakeet-tdt-0.6b-v3
uv run hf download rhasspy/piper-voices \
  pl/pl_PL/bass/high/pl_PL-bass-high.onnx \
  pl/pl_PL/bass/high/pl_PL-bass-high.onnx.json
```

OpenWakeWord feature models are not loaded from the Hugging Face cache. The initializer downloads
`embedding_model.onnx`, `melspectrogram.onnx`, and
`silero_vad.onnx` from the OpenWakeWord v0.5.1 release into
`models/openwakeword`.

Model weights and voices are separate works with their own licenses. Check the model card or download source before
redistribution or commercial use.
