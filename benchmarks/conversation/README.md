# Conversation benchmarks

These benchmarks compare routing and language-model latency without requiring a
microphone or speaker. The Polish suite is the only project file containing Polish
benchmark utterances; implementation, reports, and documentation remain in English.

Select the adapter in `settings.yml`, make the requested profile contain model
parameters for that adapter, and run:

```bash
uv run python -m tools.conversation_benchmark
```

The same command runs the current LangChain baseline or an MLX candidate. Do not
keep both providers' model identifiers in one profile.

Use `pl-henry.yml` and `pl-alexa.yml` for profile-specific blind reviews. Henry
should remain useful and stop joking in sensitive situations. Alexa should use
child-friendly language, clearly separate fact from fiction, and protect the
child's privacy without becoming frightening or patronizing.

The tool never downloads models explicitly. The selected adapter may resolve a
missing model through its underlying library, so prepare the models before running
an offline benchmark.

Review the generated Markdown report together with `results.json`. Record warm and
cold runs separately. Final acceptance also requires an end-to-end voice session,
because model-only latency does not include transcription, phrase segmentation,
synthesis, playback, or interruption.

Human reviewers should score each spoken response from 1 to 5 for correctness,
naturalness, brevity, personality consistency, and conversational appropriateness.
Compare randomized recordings produced with the same TTS voice.

Initial acceptance targets for the M1 Max are a fast first speakable phrase within
0.8 seconds of final transcription, a waiting reaction within 0.6 seconds, a first
substantive detailed phrase within 1.8 seconds, at least 90% routing accuracy, and
an average human naturalness score of at least 4 out of 5. Treat these as provisional
until the first repeatable hardware run establishes realistic baselines.
