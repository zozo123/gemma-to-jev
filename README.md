# Gemma System One

Turn a small Gemma into a language-conditioned decision function, in native
Rust — and use it as a laboratory for **semantics-preserving shared-state
inference**. This demo uses `google/gemma-3-4b-it` (Q4_K_M) with Candle on
Apple Metal.

Project page: <https://zozo123.github.io/gemma-to-jev/>

> **Measured: 47 ms per warm decision on the experimental packed-sheet path,
> 21.2 decisions/s, 7/7 option-order stability — on an Apple M1 Pro. 58.9% on
> JevBench's 231 public decisions.**
>
> The important caveat is now first-class: the packed-sheet optimization can
> change an answer. The public API therefore defaults to an isolated
> shared-prefix batch, while the faster sheet topology remains experimental.

Instead of:

```text
prompt -> generate tokens -> parse text
```

it does:

```text
prompt -> legal-label logits -> softmax -> typed answer
```

The decision path has no open-ended generation, generated JSON, or parser.

## Run

```bash
cargo run --release                 # conference demo (default)
cargo run --release -- demo
cargo run --release -- bench        # warm latency, baseline, stress
cargo run --release -- isolation-bench --repeat 10
cargo run --release -- serve        # local TypeSafe-compatible API
```

The first run downloads about 2.3 GB of weights and the tokenizer from Hugging
Face. Later runs use the local Hugging Face cache.

```bash
cargo run --release -- bench --repeat 5
cargo run --release -- bench --skip-stress
cargo run --release -- bench --skip-baseline
cargo run --release -- demo --temperature 1.5
cargo run --release -- --cpu demo
```

Temperature is an inference control, not calibration.

See [the landscape](docs/landscape.md) for related systems and
[the semantic-isolation plan](docs/semantic-isolation.md) for the research
direction.

## Use it inside an agent harness

`serve` loads Gemma once and exposes `POST /v1/systemone`, matching the
TypeSafe request/response shape. It binds to `127.0.0.1:8080` by default.

```bash
cargo run --release -- serve

curl http://127.0.0.1:8080/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "The build failed with: ld: library not found for -lssl",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Choose the next owner.",
        "criteria": {
          "build": "Build or linker failure",
          "security": "Security incident",
          "remote_llm": "Needs deeper investigation"
        }
      }
    }
  }'
```

The intended production shape is a local sidecar, not an in-process benchmark
dependency:

```text
event -> local System One -> typed route / policy / escalation decision
                             | high-trust allowlisted case: act locally
                             ` otherwise: call the remote LLM
```

Run it **before** a remote model when saving calls matters. Run both
speculatively when latency matters, but cancellation may not save provider cost.
Do not gate on this model's raw confidence alone: JevBench shows that it is
overconfident. Gate only task families validated for the application, and send
ambiguous, adversarial, long-policy, or multi-hop work to the remote model.

Single-question requests use the direct logit path. Multi-question API requests
share only the state prefix and evaluate each question in an isolated batch row.
The packed question-sheet/pointer path is retained for experiments because it is
faster but has already shown semantic drift on the demo workload.

## Result at a glance

```text
full prompt                         ~400 ms / decision
shared-state KV cache                 173 ms / decision
state + question sheet cache          108 ms / decision  [experimental]
batched 2-token sheet pointers         47 ms / decision  [experimental]
```

The 47 ms path is not a cherry-picked smaller model. It is the same quantized
Gemma 3 4B. But it is also **not semantically equivalent by assumption**:
`bench` exposes disagreements against the independent/full-prompt path, and
`isolation-bench` compares all execution topologies explicitly.

## Question primitives

Following Jev's interface shape, every question is one of three types, and each
answer carries the full restricted distribution:

| Primitive | Question | Returns |
| --- | --- | --- |
| `noul` | Is this true? | probability of yes |
| `choice` | Which of these options? | winning label, probabilities, confidence |
| `score` | Where on this scale? | fractional level, probabilities, confidence |

Confidence here is the largest probability in the restricted distribution. That
is our own shape statistic, not Jev's calibrated confidence.

## How it works

```text
                  GEMMA 3 4B

state + question + runtime labels
                |
                v
        +---------------+
        | transformer   |
        +---------------+
                |
                v
      next-position logits
                |
                v
          [A] [B] [C] [D]
                |
                v
             softmax
                |
                v
       0.04 0.87 0.07 0.02
                |
                v
            DEPENDENCY
```

Each label must be exactly one token at the real answer boundary. The program
verifies this at startup and refuses labels that tokenize any other way.

## Semantic isolation is now the research question

The basic restricted-logit trick is easy to reproduce with many open models.
The harder systems problem is to reuse one expensive state across many
runtime-defined questions **without changing what each question means**.

Reference execution asks each question independently:

```text
state + Q1 -> P1
state + Q2 -> P2
state + Q3 -> P3
```

The production multi-question path shares only the state computation:

```text
             shared state KV
             /      |      \
           Q1       Q2      Q3
           |        |       |
          P1       P2      P3
```

The experimental sheet path instead puts every question in one context and then
uses tiny answer pointers. That can be faster, but the conditioning function is
different:

```text
P_sheet(y_i) = P(y_i | state, Q1, Q2, ..., Qn, pointer_i)
```

rather than:

```text
P_ref(y_i) = P(y_i | state, Qi)
```

On the existing demo, `retry_risk` changed materially between these paths.
That failure is useful: optimization should be evaluated as a
**latency/throughput vs semantic-drift frontier**, not by speed alone.

Run:

```bash
cargo run --release -- isolation-bench --repeat 10
```

It reports selected-label flips, maximum probability drift, mean
Jensen-Shannon divergence, and end-to-end latency for independent, isolated
shared-prefix, and packed-sheet execution.

## Landscape

The space has already split into several distinct technical directions:

| Direction | Examples | What they optimize |
| --- | --- | --- |
| **Purpose-trained decision models** | Jev, [Laya](https://github.com/he-jev/laya), [Kev](https://github.com/jaredpalmer/kev), [Mapika Decider](https://github.com/Mapika/decider) | train representations/readouts for bounded decisions, calibration, and sometimes explicit question isolation |
| **Inference-time adapters** | [OpenJev](https://github.com/lookski/openjev), [SemIf](https://github.com/TheoLeeCJ/SemIf), [openjev-sglang](https://github.com/ekzhang/openjev-sglang) | reuse ordinary LMs and read restricted logits instead of generating prose |
| **Probability quality** | [AnyJev](https://github.com/nokia-applied-research/AnyJev), Kev-style calibration | remove option/label bias and turn scores into empirically useful probabilities |
| **Parallel answer architectures** | [djev / DiffusionGemma](https://github.com/mmastrac/djev) | make several bounded outputs native rather than sequential |
| **Shared-state inference systems** | vLLM prefix caching, Hydragen, DeFT | amortize expensive prefixes and branch efficiently |
| **This repo** | Gemma + Rust + Candle | measure the throughput/semantic-isolation frontier for many runtime-defined decisions over one state |

The basic pattern

```text
ordinary LM -> legal option logits -> restricted softmax -> typed answer
```

is now common. The research question here is what happens **after** that:
how much shared computation can we introduce before the optimized execution
stops computing the same decision distribution?

Full map: [docs/landscape.md](docs/landscape.md).

## Compared with Jev

This repo copies Jev's **interface** (noul / choice / score, restricted softmax, no
`generate()`). It does not copy Jev's **model**.

| | This demo | TypeSafe Jev |
| --- | --- | --- |
| What it is | Prompted Gemma 3 4B Q4, logits over A/B/C… | A hosted System One model trained with RLCD |
| Output | Typed value + raw softmax over legal labels | Typed value + calibrated probabilities |
| Many questions, one state | API: shared state prefix + isolated batch rows; experimental packed-sheet path for speed studies | Evaluate questions in parallel against one state |
| Confidence | Max probability in the restricted set | Shape statistic from a model trained to be calibrated |
| Latency | **47 ms per decision warm**, on an M1 Pro | TypeSafe quotes ~70–500 ms end-to-end, most around 100 ms, from US West |
| Throughput | **21 decisions/s** after one prefill | The ~100 ms figure is about 10 requests/s on their hardware |

The systems opportunity is **pay for the state once**. Three experiments moved
the 4B model from roughly 2.5 to 21 decisions/s, but they do not all preserve
the same semantics:

1. **Cache the state.** Prefill the common state once and run each question
   suffix against that exact prefix. 400 ms → 173 ms per decision in the
   measured serial path.
2. **Batch isolated suffixes.** Widen the state cache to independent rows and
   keep each question text in its own row. This is now the production API
   topology; `isolation-bench` checks it against independent prompts.
3. **Cache the questions too (experimental).** Put state and the whole question
   list in one prompt, then answer via tiny pointers. This reached 108 ms serial
   and **47 ms batched**, but changed at least one demo answer. It is therefore
   a speed experiment, not the semantic reference.

Two things that looked promising and did not work:

- **Batching the long suffixes** (the full question text per row) gave nothing:
  178 ms per decision versus 165 ms serial. At ~31 tokens per row the pass is
  compute-bound, so extra rows cost extra work. Batching only pays once the
  suffix is short enough to be weight-bound.
- **A one-token pointer** (`1` instead of `1.`) hit 27 ms per decision and
  destroyed the answers: labelled accuracy 1/4, every state classified
  `compilation`. The model could no longer tell the questions apart. Speed that
  fails the checks is not speed.

## Measured on an Apple M1 Pro

Warm, 4-bit 4B model, `cargo run --release -- bench --repeat 2 --skip-baseline`:

| Metric | Result |
| --- | --- |
| Model load (warm page cache) | 5.2 s |
| First inference after load | 2.5–4.4 s |
| Independent decision (full prompt) | ~400 ms · ~2.5/s |
| Shared-state suffix after prefill | p50 173 ms · 6.0/s |
| Prefilled sheet, serial | p50 108 ms · 9.2/s |
| **Prefilled sheet, batched** | p50 **47 ms · 21.2 decisions/s** |
| Sheet prefill (state + 5 questions) | 232 tokens · ~0.6 s, paid once |
| Determinism | 5/5 identical distributions |
| Option-order stability | 7/7 rotations kept the same decision |
| Labelled spot-check, full prompt | 3/4 |
| Labelled spot-check, fast sheet path | 4/4 |
| System One vs `generate()` | 420 ms / 0 tokens vs 509 ms / 2 tokens |

The fast path is checked against the slow one on every `bench` run, not assumed,
and it does not fully agree. Four of the five discrete decisions match across the
full-prompt, serial-sheet, and batched-sheet paths. The fifth, the `retry_risk`
score, lands at 3.00 on the full prompt and 1.08 on the sheet — high risk versus
low risk on a 0–4 scale, which is a different answer, not rounding. Reading a
question from a cached sheet is therefore not equivalent to asking it on its own,
and anything depending on that question should use the full-prompt path.

The remaining drift between serial and batched sheet reads is small (1.08 versus
1.14 on the same score). Single-position steps skip the sliding-window mask,
which is what upstream Candle already does when decoding, and is the likely
source of that numeric difference.

On the four-case labelled spot-check the two paths also disagree in the other
direction: the sheet path gets 4/4 while the full prompt gets 3/4, missing an
`infrastructure` case it calls `test`. Four cases decide nothing; both numbers
are too small to rank the paths.

The generation baseline emits only two tokens, so the gap is modest and varies
between runs (509–804 ms observed). The saving grows with longer outputs; the
structural win is that there is no text to parse and the output space cannot go
out of range.

### JevBench: all 231 public decisions

The TypeSafe-compatible server was run through the official
[fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) v1.3.0
adapter, serially, on every redistributable public task. These are fresh-state
requests, so they measure a different workload from the 47 ms shared-state
batch above.

| Metric | Result |
| --- | --- |
| Accuracy | **58.9%** · 136/231 |
| Easy | **100%** · 48/48 |
| Standard public | **62.5%** · 45/72 |
| Hard public | **38.7%** · 43/111 |
| Macro family accuracy | 55.3% |
| ECE / Brier | 0.402 / 0.808 |
| API and schema validity | 231/231 |
| Fresh-state latency | p50 593 ms · p95 16.6 s |

On the exact same 231 public task IDs, JevBench's published outcomes are Laya
58.4%, kev-4B 66.2%, and Jev 1.13.0 86.6%. This prompted Gemma barely clears
Laya, but it is not competitive with a trained decision model. The p95 is
dominated by long-context policy and multi-hop cases.

This is not an official leaderboard rank: no hidden set was run, and hardware
differs. The compact result artifact is in
[`bench/results/jevbench-v1.3-public.json`](bench/results/jevbench-v1.3-public.json).
Reproduce it while `serve` is running:

```bash
./bench/run_jevbench.sh
```

Two findings worth keeping:

- An earlier prompt contained a policy hint about retries. It pushed the model
  toward one answer and dropped option-order stability to 20% and the spot-check
  to 3/4. Removing it restored 100% and 4/4.
- The very first load on a cold page cache took 19 s, and early decisions took
  seconds before the weights became resident. Quote warm numbers only.

## Limitations

- Raw softmax values are model-relative scores, not calibrated probabilities.
  The model frequently reports 100%, which reflects a peaked distribution rather
  than certainty about the world.
- On the demo incident, the model answers `yes` to retrying an unchanged
  command, which is wrong for a deterministic missing library. Option-order
  testing confirms this is a genuine model judgment, not position bias.
- It routes a missing `-lssl` to the security team, which is defensible but
  debatable.
- The fast path needs the question list up front, which is what makes the
  pointer work. Questions discovered mid-flight fall back to the 173 ms
  shared-state path.
- The 47 ms figure is the warm batched pass. The sheet prefill (~0.6 s) is paid
  once per state, so a single question against a fresh state is not fast.
- This batches rows itself rather than being Jev's parallel sampler, and the
  model is still a prompted chat model rather than one trained for decisions.

## Next

The priority is no longer "make another Jev wrapper." It is to map and then
push the **semantic-isolation frontier**:

1. expand `isolation-bench` across question counts, state lengths, order
   permutations, distractors and public benchmark tasks;
2. optimize the isolated shared-prefix path without relaxing the semantic
   contract (cache reuse, suffix bucketing, branch-aware attention, better
   kernels);
3. add held-out calibration/debiasing experiments; current JevBench ECE is
   0.402;
4. compare raw final-token logits against intermediate-layer / trained decision
   readouts;
5. only after the text contract is stable, add shared multimodal state so one
   image/world observation can feed many isolated decisions.

Detailed plan: [docs/semantic-isolation.md](docs/semantic-isolation.md).

Landscape and related work: [docs/landscape.md](docs/landscape.md).
