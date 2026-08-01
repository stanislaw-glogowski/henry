# Engineering Guidelines

This document defines the default engineering conventions for Henry. It applies
to production code, tests, native audio code, build tooling, and documentation.

The terms below express intent:

- **Must** indicates a required invariant.
- **Should** indicates the default choice; departures require a concrete reason.
- **Prefer** indicates a consistency or readability convention.

Task-specific user instructions take precedence. Preserve unrelated work and do
not broaden a change merely to make surrounding code match these guidelines.

## Design principles

- Prefer explicit, conventional code over additional abstraction layers.
- Introduce an abstraction only for a real ownership boundary, external
  dependency, concurrency boundary, or multiple meaningful implementations.
- Choose names from the problem domain rather than the implementation
  technology.
- Keep refactors mechanical unless behavior changes are explicitly in scope.
- Do not design prematurely for distributed execution, remote providers, or a
  plugin system.
- Optimize for a readable voice pipeline and predictable resource ownership.

## Package boundaries

- `henry_speech` must own audio, VAD, wake-word detection, utterance
  segmentation, STT, TTS, playback, and voice-session transitions.
- `henry_conversation` must own conversation inputs, LangGraph state and
  routing, history, summaries, response generation, and reply segmentation.
- `henry_resources` must own discovery and loading of profiles, prompts,
  settings, and model paths.
- `henry_common` must contain only genuinely shared lifecycle, event, logging,
  and validation primitives.
- `henry_cli` must remain the application composition root. Feature packages
  may assemble their internal services behind explicit runner functions.
- A package must not make decisions owned by another domain. Speech may publish
  `ConversationActivated` or `UserTurn`; it must not choose graph nodes.

## Module layout

- Keep related immutable value objects and type aliases in `domain.py`.
- Keep validated profile and runtime settings in `config.py`.
- Keep minimal consumer-facing abstractions in `ports.py`.
- Keep external-library and operating-system implementations below `adapters/`.
- Give stateful helpers and algorithms their own semantic modules, such as
  `buffer.py`, `resampler.py`, or `segmenter.py`.
- One element per file is not an absolute rule. Small, tightly related domain
  values should stay together; a stateful component should normally stand
  alone.
- A local `helpers.py` is acceptable only for a narrow set of technical helpers.
  Split it when a distinct responsibility emerges.
- Prefer noun-based filenames that match their primary element: `resampler.py`,
  not `resampling.py`.

## Public package APIs

- Treat package `__init__.py` files as explicit public facades.
- Export public names through a sorted `__all__`.
- Do not expose adapter internals, wire helpers, or implementation-only types.
- Keep adapter subpackage `__init__.py` files empty unless they provide a real,
  stable public API.
- Do not use module-level `__getattr__` solely to work around eager imports.
- Import public domain contracts from package facades. Factories should import
  concrete adapters directly from their implementation modules.
- Avoid chains of wrapper functions that only rename or reexport another
  function.

## Domain models and ports

- Domain value objects should use `@dataclass(frozen=True, slots=True)`.
- Domain modules must not import concrete adapters.
- Domain values may contain small operations that protect their invariants or
  construct closely related values.
- Stateful algorithms do not belong in `domain.py`.
- A port must contain only operations required by its consumer.
- Do not add device metadata to `AudioInput` when it belongs to `AudioDriver`.
- Do not leak external-library types through public domain ports.
- Use abstract resource classes for ports that have explicit `open()` and
  `close()` lifecycle.
- Use the Python 3.14 `type` statement for type aliases.
- Use `Enum` with `auto()` for internal states. Use explicit stable values for
  wire formats and persisted contracts.

## Naming

- Name classes after a responsibility or domain concept, not a generic action.
- Add a domain prefix when a public name would otherwise be ambiguous, for
  example `AudioPlaybackOutcome`.
- `Service` means a component that manages an operation and resource lifecycle.
- `Worker` means long-running coordination across queues, services, or events.
- `Driver` means ownership of a device session or shared hardware process.
- `Input` and `Output` describe operational streams, not device metadata.
- `Model` describes an inference port or its concrete adapter.
- Name operations with verbs: `read`, `write`, `detect`, `transcribe`,
  `synthesize`, and `interrupt`.
- Name stable owned values with nouns and expose them as properties: `input`,
  `output`, `devices`, and `format`.
- Properties must not hide expensive I/O or resource creation.
- Use `is_*`, `has_*`, or `can_*` for predicates.
- Keep independent constants separate even when their current values match.
- Put a constant in its class when it describes only that class. Put it at
  module scope when multiple module elements share it.

## Adapter factories and imports

- Adapter factories should remain simple `match` statements over validated
  configuration.
- Import only the selected concrete adapter inside its factory branch.
- Do not introduce registries, dependency containers, or plugin discovery while
  a direct factory remains sufficient.
- A package facade may reexport its adapter factory directly; it should not wrap
  it without adding behavior.
- Ordinary imports of Henry packages must not initialize MLX, Metal, models, or
  audio devices.
- Import heavy ML runtimes inside `open()` or a helper called by `open()`.
- Keep model loading and inference in the worker thread that owns the model.
- Use `TYPE_CHECKING` for imports needed only for annotations or to break a
  runtime cycle.
- Remember that `typing.get_type_hints()` resolves annotation names at runtime.
- Do not add `from __future__ import annotations` mechanically; Python 3.14 is
  the project baseline.
- Limit `Any` to poorly typed external-library boundaries. Do not use `cast()`
  to conceal an ownership or dependency problem.

## Resource lifecycle

- Constructors should store dependencies and initialize lightweight local
  state. They must not open devices or load models.
- `open()` should acquire expensive resources and must reject a second open.
- `close()` should be safe to call repeatedly.
- Operations called before `open()` should fail with a precise `RuntimeError`.
- Session-dependent properties should be available only while their owner is
  open.
- When opening multiple resources, register each cleanup immediately with an
  `ExitStack` so partial startup unwinds in reverse order.
- Closing a composite resource should attempt every cleanup and report or log
  all failures.
- Do not hide lifecycle transitions inside getters or properties.

## Threads, asyncio, and queues

- The asyncio event loop coordinates work; it must not perform blocking audio
  operations or ML inference.
- A blocking model must have one owning worker or executor.
- Model `open()`, inference, cancellation-sensitive mutation, and `close()` must
  run in the owning thread.
- Cross the thread/event-loop boundary with queues, futures, or
  `loop.call_soon_threadsafe(...)`.
- Capture and playback require independent execution because either may block.
- Interruption may use a separate thread when the playback executor is occupied.
- Prefer `asyncio.TaskGroup` for related long-running tasks.
- Shutdown must signal or cancel operations, await them, close owned resources,
  and only then shut down executors.
- Every successful queue `get()` must have exactly one matching `task_done()`,
  preferably in `finally`.
- Queue draining must preserve the same accounting rule.
- Define sentinel, cancellation, and overflow policies explicitly.
- Do not use a queue as hidden global state.

## Events and conversation flow

- Events must be immutable domain values.
- Name commands as actions, such as `GenerateReply` and `CancelReply`.
- Name facts and observations as completed events, such as
  `ReplyGenerationStarted` and `WakeWordObserved`.
- Telemetry must not drive domain decisions.
- Every event subscriber must account for consumed events with `task_done()`.
- `ReplyChunk` is streamed generation data; `ReplyPhrase` is complete plain text
  ready for independent synthesis.
- Confirm delivery only after all audio frames preceding a phrase boundary have
  played successfully.
- Every LangGraph invocation must be finite. The worker waits for future input;
  the graph does not.
- `ConversationActivated` routes through `opening`; `UserTurn` routes through
  `reply` and `summarize`.
- Keep mutable messages and summaries in graph state. Keep model settings and
  prompts in immutable runtime context.
- Pass interrupted-delivery context into the next graph run. Do not mutate an
  active checkpoint while cancelling its run.

## Audio and the native helper

- Every `AudioFrame` must carry an `AudioFormat`.
- Device buffer size, transport frame size, and inference frame size are
  separate concepts even when their values currently match.
- Resampling and buffering must explicitly bridge native and exposed formats.
- Name adapter output contracts by what the adapter emits, not by the current
  downstream model.
- `AVFAudioDriver` owns one full-duplex native process. Its input and output
  ports do not own separate device sessions.
- Voice Processing must receive Henry playback as its echo-cancellation
  reference.
- Wire payloads must be structured values, not text parsed from log messages.
- A required wire message or semantic change must increment the protocol
  version.
- Protocol changes must update Python, Swift, the fake helper, and tests in the
  same change.
- Device information should originate in the layer that actually selected the
  devices.

## Configuration and resources

- Configuration models must be frozen and reject unknown fields.
- Profiles describe persona and profile-dependent choices. Settings describe
  technical runtime selection and tuning.
- Use `driver` for the audio-device implementation and `adapter` for model
  implementations.
- Validate configuration when it is loaded, not during inference.
- Keep the profile prompt layout fixed. Do not add configurable prompt paths.
- `henry_resources` must not assume the process working directory.
- Resolve local data through `HENRY_HOME`, the nearest `.henry`, and then the
  platform data directory.
- Never package or commit downloaded models, recordings, caches, secrets, or
  user-specific runtime data.

## Errors and logging

- Use `ValueError` for an invalid argument or configuration value.
- Use `RuntimeError` for invalid lifecycle, adapter state, or protocol behavior.
- Use `FileNotFoundError` for a missing required resource.
- Error messages should identify the owning layer and, when useful, the expected
  and actual values.
- Preserve lower-level causes with `raise ... from error`.
- Do not use `assert` to validate user data or runtime state.
- Use the component-bound logger.
- Log important resource open, ready, and close transitions.
- Log selected devices and an audio format once, not for every frame.
- Use `print()` only in deliberately interactive tools such as diagnostics and
  benchmark recording.
- Do not log secrets, complete prompts, or unbounded conversation history.

## Comments, language, and documentation

- Document public contracts, significant constants, thread ownership, and
  non-obvious lifecycle or control flow.
- Comments should explain why, not restate what the next line does.
- Do not add docstrings to trivial getters or self-explanatory implementations.
- Project code, errors, logs, comments, and documentation must be in English.
- Polish is appropriate for benchmark source material, text intended to be
  spoken, and clearly marked language examples.

## Testing

- Mirror source packages under `tests/` and directly cover executable modules or
  public contracts.
- Test domain and service behavior through fake ports.
- Put reusable fake ports and resources in `tests/support.py`.
- Automated tests must not require microphones, speakers, Metal, models,
  downloads, Ollama, or real PyAudio devices.
- Test normal lifecycle, partial startup, runtime failure, cancellation, and
  shutdown.
- Synchronize concurrent tests with events, queues, or futures and protect waits
  with `asyncio.wait_for(...)`.
- Do not use arbitrary sleeps as a synchronization mechanism.
- Concrete adapters may be excluded from coverage, but lightweight contract
  tests should mock external libraries where useful.
- Test Swift wire and value behavior independently from hardware.
- Report microphone, speaker, AEC, Metal, and model smoke tests separately from
  automated verification.
- Do not make an STT or TTS candidate the default before a recorded Polish
  benchmark; TTS changes also require blind listening review.

## Dependencies, build, and change discipline

- Put runtime dependencies in `project.dependencies`.
- Put test, lint, and IDE tooling in the development dependency group.
- Put isolated backend requirements in `build-system.requires`.
- A build tool may also be a development dependency when locally maintained
  hook code imports it and the IDE analyzes that code.
- The native helper must be included in the wheel, and the wheel must carry the
  correct macOS ARM64 platform tag.
- A native source change must cause the helper to rebuild.
- Inspect staged and unstaged changes before editing and preserve unrelated
  user work.
- Update an affected contract, implementation, fakes, tests, and documentation
  together.
- Run the complete repository verification defined in `AGENTS.md` before
  reporting completion.
- State which hardware or model checks remain unverified.
