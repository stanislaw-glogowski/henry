# AGENTS.md

## Scope

These instructions apply to the entire repository.

Henry is a local Apple Silicon voice assistant. Preserve the explicit boundaries between domain code, ports, adapters,
services, and orchestration. Prefer small, conventional contracts over additional abstraction layers.

## Environment

- Python: 3.14
- Package manager: uv
- Runtime target: macOS on Apple Silicon
- Application entrypoint: `henry-cli` (`-noui` / `--no-ui` for console logging)
- Local data: `HENRY_HOME`, the nearest `.henry/`, or the platform data directory

Use the following verification commands:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff check src tests
UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff format --check src tests pyproject.toml
UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest -q
UV_CACHE_DIR=/private/tmp/uv-cache uv run python -m compileall -q src tests
```

Use `uv sync` after dependency or lockfile changes.

## Architecture boundaries

- `henry_client.audio` owns audio formats, streams, VAD, wake-word analysis, and audio input/output workers.
- `henry_client.speech` owns utterance segmentation, STT, and TTS.
- `henry_client.reply` owns language-model response generation and reply segmentation.
- `Orchestrator` owns product flow and listening-mode transitions.
- `henry_cli_old` owns Textual UI state, console diagnostics, and telemetry presentation, not domain decisions. Do not
  restore a separate debugger entrypoint.
- `henry_common` resolves local files; it must not assume the process cwd.

The assistant remains in utterance mode for follow-ups after wake-word activation. `max_empty_segments` controls when
consecutive empty utterance timeouts return it to wake-word mode; a completed utterance resets the counter.

Blocking adapters and ML runtimes must remain in their owning worker threads. Cross the thread/event-loop boundary with
`queue.Queue` and
`loop.call_soon_threadsafe(...)`. Never mutate a thread-owned model directly from the event loop.

Each successful queue `get()` has exactly one matching `task_done()`, preferably in `finally`. Shutdown must cancel and
await asyncio tasks before service contexts join their workers.

## Tests

- Keep unit and component tests independent of microphones, speakers, Metal, ONNX files, Hugging Face downloads, and
  real PyAudio devices.
- Use fake implementations of ports with real asyncio queues and threads.
- Use Textual's headless test support for CLI UI behavior.
- Synchronize with concrete events, queues, or futures and protect awaits with
  `asyncio.wait_for(...)`; do not use arbitrary sleeps.
- Cover both normal lifecycle and startup/runtime error propagation.
- Treat a manual device smoke test as separate evidence from automated tests.
- `uv run pytest -q` enforces at least 95% combined coverage with branch measurement for `henry_cli_old`,
  `henry_speech`, and
  `henry_common`.
- Concrete modules below `*/adapters/*` are excluded from the coverage threshold, but focused adapter contract tests may
  still run.

Place tests under `tests/` mirroring the source package when practical. Shared fake ports and resources belong in
`tests/support.py`.

## Change discipline

- Preserve unrelated user changes and inspect both staged and unstaged state.
- Do not change wake-word session semantics, timing, model selection, prompts, or audio formats as incidental cleanup.
- Keep comments and docstrings minimal. Document public API contracts, non-obvious constants, and thread ownership; do
  not narrate straightforward implementation.
- Keep spoken assistant output plain text and compatible with line-buffered TTS.
- Do not commit `.henry`, downloaded models, caches, generated audio, or secrets.

Before reporting completion, run the complete automated verification and list any manual hardware checks that remain
unverified.
