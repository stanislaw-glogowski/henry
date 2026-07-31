# AGENTS.md

## Scope

These instructions apply to the entire repository.

Henry is a local Apple Silicon voice assistant. Preserve explicit boundaries between conversation, speech, resources,
shared infrastructure, and the CLI composition root. Prefer small, conventional contracts over additional abstraction
layers.

## Environment

- Python: 3.14
- Package manager: uv
- Runtime target: macOS on Apple Silicon
- Application entrypoint: `henry-cli` or `python -m henry_cli`
- The CLI has no runtime arguments. It loads the `default` profile and logs to the console.
- Local data: `HENRY_HOME`, the nearest `.henry/`, or the platform data directory

Use the following verification commands:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff check src tests
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff format --check src tests pyproject.toml
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run python -m compileall -q src tests
UV_CACHE_DIR=/private/tmp/uv-cache uv build
```

Use `uv sync` after dependency or lockfile changes.

## Architecture boundaries

- `henry_speech` owns audio formats and streams, VAD, wake-word analysis, utterance segmentation, STT, TTS, playback,
  and voice-session listening transitions.
- `henry_conversation` owns conversation inputs, LangGraph state and routing, conversation summaries, language-model
  response generation, and reply segmentation.
- `henry_resources` resolves and loads local profiles, prompt files, settings, and model paths. It must not assume the
  process cwd.
- `henry_cli` is the only application composition root. It selects defaults, configures console logging, installs signal
  handlers, and starts workers. It does not make conversation or speech decisions.
- `henry_common` owns shared lifecycle, event-bus, logging, and validation primitives.

One wake-word detection activates a voice session and follow-up utterances do not require another wake word. Speech
publishes either `ConversationActivated` or `UserTurn`; it does not choose graph nodes. Conversation runs are finite:
activation routes to `opening`, while a user turn routes through `reply` and `summarize`. The graph never waits for the
next utterance.

Conversation history and its summary are mutable graph state. Profile-derived model settings and prompts are immutable
runtime context. The local CLI uses an in-memory checkpointer and `thread_id="default"`, so history lasts only for the
current process.

The profile contract is fixed:

```text
profiles/<profile-id>/
  profile.yml
  prompts/
    system.md
    opening.md
    summary.md
```

Do not add configurable prompt paths. Keep persona, opening behavior, summary instructions, and other profile-specific
text in these files rather than Python constants.

Blocking adapters and ML runtimes must remain in their owning worker threads. Cross the thread/event-loop boundary with
`queue.Queue` or `loop.call_soon_threadsafe(...)`. Never mutate a thread-owned model directly from the event loop.

Each successful queue `get()` has exactly one matching `task_done()`, preferably in `finally`. Shutdown must cancel and
await asyncio tasks before service contexts join their workers.

## Tests

- Mirror every source package under `tests/` and directly cover each executable module or its public contract.
- Keep tests independent of microphones, speakers, Metal, ONNX files, Hugging Face downloads, Ollama, and real PyAudio
  devices.
- Use fake implementations of ports with real asyncio queues and threads.
- Use fake chat models for graph and streaming behavior.
- Synchronize with concrete events, queues, or futures and protect awaits with `asyncio.wait_for(...)`; do not use
  arbitrary sleeps.
- Cover normal lifecycle, cancellation, startup/runtime error propagation, profile validation, graph routing,
  checkpoint isolation, and reply line buffering.
- Treat manual device and model smoke tests as separate evidence from automated tests.
- `uv run pytest -q` enforces at least 95% combined branch coverage for `henry_cli`, `henry_common`,
  `henry_conversation`, `henry_resources`, and `henry_speech`.
- Concrete modules below `*/adapters/*` are excluded from the coverage threshold, but lightweight adapter contract tests
  should mock external libraries where useful.

Shared fake ports and resources belong in `tests/support.py`.

## Change discipline

- Preserve unrelated user changes and inspect both staged and unstaged state.
- Do not change wake-word session semantics, timing, the Ollama model, prompts, or audio formats as incidental cleanup.
- Keep comments and docstrings minimal. Document public contracts, significant constants, and non-obvious concurrency;
  do not narrate straightforward implementation.
- Keep spoken assistant output plain text and compatible with line-buffered TTS.
- Do not commit `.henry`, downloaded models, caches, generated audio, or secrets.

Before reporting completion, run complete automated verification and list manual hardware or model checks that remain
unverified.
