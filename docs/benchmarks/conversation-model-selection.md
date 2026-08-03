# Conversation model selection

Date: 2026-08-02

## Decision

Use one model for all conversation roles within each local profile:

| Profile | Model | Fast tokens | Detailed tokens | Fast temperature | Detailed temperature |
| --- | --- | ---: | ---: | ---: | ---: |
| Alexa | `mlx-community/gemma-4-26b-a4b-it-4bit` | 256 | 512 | 0.15 | 0.25 |
| Henry | `mlx-community/gemma-4-26b-a4b-it-4bit` | 384 | 768 | 0.0 | 0.0 |
| Gizmo | `mlx-community/Qwen3.6-35B-A3B-4bit` | 384 | 768 | 0.0 | 0.0 |
| Lucy | `mlx-community/gemma-4-26b-a4b-it-4bit` | 384 | 768 | 0.0 | 0.0 |
| Viki | `mlx-community/gemma-4-26b-a4b-it-4bit` | 384 | 640 | 0.0 | 0.0 |

The classifier uses the same model as the other roles with eight tokens and a
temperature of zero. A selected profile loads only its own model, so choosing Qwen
for Gizmo does not make Gemma and Qwen resident in the same Henry process.

Gemma is the best general choice: it combines the strongest spoken Polish of the
fast candidates with sub-second warm delivery of the first model chunk. Gizmo is
the exception. Gemma repeatedly produced Polish inflection errors in playful
answers, including with deterministic sampling. Qwen 3.6 35B A3B was slower but
produced substantially cleaner child-facing responses and followed the safety
contract.

## Environment

- MacBook Pro with Apple M1 Max and 64 GB memory.
- macOS 26.5.2.
- Python 3.14.6.
- MLX 0.32.0.
- MLX-LM 0.31.3.
- Repository baseline: `aa6ce94604979c846fa1073697ca129c4575207a`.

The benchmark used locally cached model snapshots:

| Model | Snapshot revision | Approximate cache size |
| --- | --- | ---: |
| `mlx-community/Qwen3.5-4B-MLX-4bit` | `32f3e8ecf65426fc3306969496342d504bfa13f3` | 3.1 GB |
| `mlx-community/Qwen3.5-9B-MLX-4bit` | `938d8919941c6e7efd3c7150eff7fe9d12afa631` | 6.0 GB |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | `890b4c43f99ff392819d83605f7b1e59fa9688aa` | 8.2 GB |
| `mlx-community/Qwen3.6-27B-4bit` | `c000ac2c2057d94be3fa931000c31723aac53282` | 16.1 GB |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | `38740b847e4cb78f352aba30aa41c76e08e6eb46` | 20.4 GB |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | `0d77464eeb233a2da68ebf9d7dc4edaac7db956d` | 15.4 GB |
| `mlx-community/gpt-oss-20b-OptiQ-4bit` | `7f962b6de641701a4403ccc910055ac54ae7e72e` | 11.7 GB |
| `speakleash/Bielik-11B-v3.0-Instruct-MLX-4bit` | `4e62550d51d2e3f61e13d3752c25f630277f53a2` | 6.3 GB |

## Method

The evaluation had four stages:

1. A two-case compatibility smoke test for all eight candidates.
2. The five-case `pl-core.yml` suite for every compatible candidate.
3. All five persona suites for Gemma, Qwen 3.6 35B A3B, Qwen 3.5 9B OptiQ,
   and Bielik 11B.
4. Prompt and parameter tuning followed by control runs of the selected models.

Every run used the production MLX adapter and a real Metal model load. The first
case in a run is cold; subsequent cases reuse the loaded model and are warm. The
reported first-chunk latency starts before optional routing classification and ends
when the adapter yields its first non-empty model chunk.

Raw generated output remains below `.henry/benchmarks/conversation` and is not
committed. The tables below contain the decision-relevant aggregate results.

## Compatibility screening

| Candidate | Cold first chunk | Warm first chunk | Warm total | Outcome |
| --- | ---: | ---: | ---: | --- |
| Qwen 3.5 4B | 3.763 s | 2.130 s | 3.110 s | Rejected: recurrent Polish and factual errors. |
| Qwen 3.5 9B | 5.360 s | 3.820 s | 5.251 s | Rejected: slower and still linguistically unstable. |
| Qwen 3.5 9B OptiQ | 6.244 s | 3.817 s | 7.343 s | Advanced: better than the ordinary 9B variant. |
| Qwen 3.6 27B | 17.085 s | 13.015 s | 18.471 s | Rejected: latency is unsuitable for voice conversation. |
| Qwen 3.6 35B A3B | 7.040 s | 2.152 s | 3.923 s | Advanced: good quality at moderate latency. |
| Gemma 4 26B A4B | 6.942 s | 0.357 s | 2.076 s | Advanced: best initial speed and spoken format. |
| Bielik 11B | 7.620 s | 0.277 s | 6.617 s | Advanced as the Polish reference. |
| GPT-OSS 20B OptiQ | 6.269 s | 0.266 s | 2.854 s | Rejected: raw Harmony analysis and channel markers leaked into speech output. |

The GPT-OSS result is an adapter incompatibility, not a conclusion about the base
model's language quality. It requires explicit Harmony channel handling before it
can be evaluated fairly in Henry.

## Shared-suite comparison

| Candidate | Cold first chunk | Warm first chunk | Warm total | Mean characters |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.5 4B | 3.768 s | 2.132 s | 3.862 s | 417 |
| Qwen 3.5 9B | 5.880 s | 3.858 s | 6.731 s | 428 |
| Qwen 3.5 9B OptiQ | 6.215 s | 3.863 s | 7.476 s | 481 |
| Qwen 3.6 27B | 16.784 s | 13.214 s | 18.411 s | 271 |
| Qwen 3.6 35B A3B | 6.607 s | 2.168 s | 3.617 s | 266 |
| Gemma 4 26B A4B | 6.279 s | 0.421 s | 2.298 s | 321 |
| Bielik 11B | 7.610 s | 0.402 s | 6.198 s | 473 |

All candidates recorded 60% routing accuracy because routing is owned by Henry's
`ResponseRouter`, not by the selected response model. With
`classify_ambiguous: false`, `detailed-001` and `planning-001` were sent to the
fast role even though the suite expects detailed responses. This is a confirmed
router or suite-contract issue and must not be used to rank models.

The qualitative failures were more decisive than small latency differences:

- Qwen 3.5 4B produced malformed Polish and invented an incoherent four-day plan
  for a weekend request.
- Both Qwen 3.5 9B variants produced frequent agreement and word-choice errors;
  the OptiQ persona run also emitted the Spanish word `orgullo`.
- Qwen 3.6 27B was generally coherent but its warm first chunk took about thirteen
  seconds.
- Bielik was strong in Polish but routinely verbose, reached token limits, emitted
  Markdown or emoji despite the spoken-output contract, and sometimes continued
  beyond the useful answer.
- Qwen 3.6 35B A3B was concise and capable, but slower than Gemma and showed
  occasional Polish errors in the broader persona run.
- Gemma gave the most consistently natural Polish and best adherence to short,
  phrase-streamed output.

## Persona finalist comparison

The following numbers aggregate 21 persona cases per candidate.

| Candidate | Mean cold first chunk | Mean warm first chunk | Mean warm total | Mean characters |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 26B A4B | 4.607 s | 0.398 s | 1.909 s | 260 |
| Qwen 3.6 35B A3B | 4.813 s | 1.989 s | 4.093 s | 294 |
| Qwen 3.5 9B OptiQ | 5.236 s | 3.662 s | 5.829 s | 299 |
| Bielik 11B | 6.854 s | 0.459 s | 5.195 s | 402 |

Initial Gemma runs at temperatures from 0.5 to 0.65 occasionally produced Polish
inflection errors or ignored a serious-situation instruction. Lower but non-zero
temperatures still changed the failures between repeated runs. Henry, Lucy, Viki,
and Gizmo were therefore tested with deterministic generation. This stabilized
Henry, Lucy, and Viki, but Gemma retained repeatable language errors in Gizmo's
playful responses. A deterministic Qwen 3.6 35B A3B control run removed those
errors and became the final Gizmo choice despite its higher latency.

## Final profile results

| Profile | Model | Cold first chunk | Warm first chunk | Warm total | Review scores C / N / B / P |
| --- | --- | ---: | ---: | ---: | --- |
| Alexa | Gemma 4 | 4.695 s | 0.369 s | 0.808 s | 4.75 / 5.00 / 5.00 / 5.00 |
| Henry | Gemma 4 | 5.205 s | 0.420 s | 1.726 s | 3.67 / 4.33 / 5.00 / 4.67 |
| Gizmo | Qwen 3.6 35B A3B | 7.079 s | 2.233 s | 3.381 s | 4.75 / 5.00 / 4.75 / 4.50 |
| Lucy | Gemma 4 | 5.271 s | 0.456 s | 2.300 s | 4.67 / 5.00 / 4.00 / 5.00 |
| Viki | Gemma 4 | 6.727 s | 0.366 s | 1.986 s | 4.75 / 5.00 / 4.25 / 4.50 |

The review columns are correctness, naturalness, brevity, and personality on a
one-to-five scale. They are an unblinded single-reviewer assessment of the written
responses. The detailed per-case notes remain in the local generated reports.

Henry's lower correctness score comes from `henry-detail-001`: the response was
clear and in character but too high-level to count as a complete project plan.
Viki intentionally becomes less theatrical in sensitive situations. Her
organization prompt was tightened after a control run aimed snobbery at the user's
possessions rather than at the abstract disorder.

## Reproduction

Run one candidate without editing a profile:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache uv run python -m tools.conversation_benchmark \
  --profile default \
  --model-id mlx-community/gemma-4-26b-a4b-it-4bit \
  --suite benchmarks/conversation/suites/pl-core.yml \
  --output .henry/benchmarks/conversation/reproduction/gemma4-core
```

Use the matching profile and persona suite for persona evaluation. Model runs need
Metal access and must execute outside environments that hide the Apple GPU.

## Limitations and next measurements

This result is sufficient for selecting the current local conversation models, but
it is not an end-to-end voice benchmark:

- Most cases are single-turn and were not repeated enough for statistical claims.
- First-chunk latency is not latency to the first speakable phrase.
- The tool does not record generated token count, tokens per second, peak memory,
  finish reason, or whether generation stopped at `max_tokens`.
- The shared-suite routing result measures the current rule-based router and should
  be fixed or separated from model scoring.
- Prompt-cache availability differs by model chat template; the Qwen 3.6 run fell
  back to uncached prompt generation.
- No microphone, transcription, phrase-segmentation, TTS, playback, AEC, or
  interruption measurement was performed.
- TTS voices still require randomized recordings and a blind listening review.

The next benchmark iteration should add multi-turn scenarios, repeated seeds,
finish-reason and token telemetry, time to the first complete phrase, and a separate
end-to-end voice report. Those additions should not block use of the selected local
profiles.
