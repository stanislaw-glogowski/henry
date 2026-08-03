# Conversation benchmarks

These benchmarks measure response routing and language-model generation without opening a microphone, synthesizing
speech, or starting the Henry application. Committed suites contain Polish inputs; implementation and reports remain in
English.

The latest local model-selection decision and its limitations are documented in
[the conversation model selection report](../../docs/benchmarks/conversation-model-selection.md).

## Running a benchmark

The default command uses the `default` profile, the conversation adapter from
`settings.yml`, and `pl-core.yml`:

```bash
uv run python -m tools.conversation_benchmark
```

Select another local profile or committed suite explicitly:

```bash
uv run python -m tools.conversation_benchmark \
  --profile default \
  --suite benchmarks/conversation/suites/pl-henry.yml
```

Override all model roles for a candidate comparison without editing the profile:

```bash
uv run python -m tools.conversation_benchmark \
  --profile default \
  --model-id mlx-community/gemma-4-26b-a4b-it-4bit \
  --suite benchmarks/conversation/suites/pl-core.yml
```

Use `--output` to choose the result directory. Without it, results are written below the local Henry data root:

```text
benchmarks/conversation/<profile>-<adapter>-<UTC-timestamp>/
├── results.json
└── report.md
```

The selected profile must contain model parameters valid for the adapter in
`settings.yml`. Direct MLX-LM inference is the supplied default; LangChain is an alternative for an Ollama-compatible
endpoint. Do not keep fields unsupported by the selected adapter in one profile.

The tool does not download models explicitly. The selected adapter may resolve missing weights through its underlying
library, so prepare models before an offline run.

## Suites

- `pl-core.yml` checks general short and detailed response routing.
- `pl-henry.yml` checks the supplied Henry persona.
- `pl-alexa.yml`, `pl-gizmo.yml`, `pl-lucy.yml`, and `pl-viki.yml` support local persona comparisons; the suites do
  not install matching runtime profiles.

Persona suites should be reviewed against the profile they name. Henry should remain useful, vary his situational irony,
and stop joking in sensitive situations. Alexa should answer briefly and admit uncertainty instead of guessing. Gizmo
should be genuinely funny for a child while separating facts from fiction and protecting privacy. Viki should sound vain
and polished without belittling the user or inventing a public life. Lucy should sound warm, perceptive, and natural,
offer a real opinion when useful, use humor sparingly, and challenge weak assumptions without becoming argumentative.

## Measurements

For each case, `results.json` records:

- expected and selected response mode;
- whether the case was the first, cold run or a subsequent warm run;
- time to the first model chunk;
- total generation time;
- output length and complete text.

`report.md` aggregates routing accuracy and provides a table for human review. Score correctness, naturalness, brevity,
personality consistency, and conversational appropriateness from 1 to 5.

Time to the first model chunk is not time to the first speakable phrase. This benchmark does not measure transcription,
phrase segmentation, waiting reactions, synthesis, playback, or interruption. End-to-end latency and spoken quality
require a real voice session, and TTS comparisons require randomized recordings and a blind listening review.

Treat a routing accuracy target such as 90% as provisional until the suite is large enough and repeatable on the target
machine. Keep cold and warm results separate when comparing adapters or models.
