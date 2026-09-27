# Semantic isolation: research plan

## Question

How much throughput can a decision runtime gain by sharing computation **without
changing the semantic function being evaluated**?

For a state `S` and question `Q_i`, define the reference distribution:

```text
P_i(y) = P(y | S, Q_i)
```

An optimized topology produces:

```text
P~_i(y) = P(y | S, Q_i, execution_topology)
```

A performance optimization is useful only if the distribution drift and decision
flip rate remain inside an explicitly measured tolerance.

## Topologies in this repo

### 1. Independent prompt — reference

Each question gets a fresh full prompt and no shared KV state.

Pros: simplest semantic reference.

Cons: repeats the expensive state for every question.

### 2. Isolated shared-prefix batch — production default

The state prefix is computed once, widened to independent batch rows, and each
row receives only its own question suffix.

Conceptually:

```text
             state KV
          /     |      \
       row A   row B   row C
        Q_A     Q_B     Q_C
         |       |       |
        P_A     P_B     P_C
```

No question text is deliberately placed in another question's context. The
benchmark compares this path against independent prompts.

### 3. Packed question sheet — experimental speed path

The state and all questions are prefilled into one context. A tiny pointer such
as `1.` or `2.` selects which question to answer.

This can be much cheaper on the hot path, but it changes the conditioning
context. The repository has already observed a materially different
`retry_risk` answer on the demo workload, so this topology must not be treated
as semantically equivalent merely because it is faster.

## Benchmark

Run:

```bash
cargo run --release -- isolation-bench --repeat 10
```

The command compares all three topologies on the same question set and reports:

- selected-label flip count;
- maximum absolute probability drift, `max |Δp|`;
- mean Jensen-Shannon divergence;
- end-to-end latency including the relevant prefill.

No result should be promoted to the README as a measured number until the
command has been run on a named model/runtime/hardware configuration.

## Phase A — establish the frontier

Expand the benchmark from the five demo questions to a reproducible matrix:

- question count: 1, 2, 4, 8, 16, 32;
- state length buckets;
- option count;
- question order permutations;
- unrelated distractor questions;
- paraphrases;
- adversarially correlated questions;
- JevBench-compatible public tasks where multi-question grouping is meaningful.

Measure:

```text
latency
throughput
peak memory
selected-label flips
total-variation / JS distance
calibration drift
option-order sensitivity
question-order sensitivity
```

## Phase B — make the isolated path faster

Keep the semantic contract fixed and optimize underneath it:

1. avoid rebuilding widened caches when the same state is reused;
2. bucket/pad suffixes more efficiently;
3. explore block-causal / branch-isolated attention where supported;
4. compare the current row-per-question implementation with tree/prefix-serving
   approaches inspired by Hydragen, DeFT, vLLM and SGLang;
5. optimize quantized matmul / Metal kernels only after semantic equivalence is
   continuously measured.

The objective is:

```text
maximize throughput
subject to
  selected-label flip rate <= epsilon
  distribution drift <= delta
```

rather than "minimize milliseconds at any semantic cost."

## Phase C — model quality

Execution speed cannot repair a weak decision model. Separate future axes:

- temperature / isotonic / conformal calibration on held-out decisions;
- AnyJev-style option-order and label-prior debiasing;
- intermediate-layer readout experiments;
- LoRA / decision-head training;
- compare prompted Gemma against Kev/Laya/Decider-style trained models.

## Phase D — multimodality

After the text semantics benchmark is stable:

- add image/attachment state support;
- encode visual state once;
- fan out many isolated questions over the shared multimodal representation;
- compare against wrappers that resend the same image per question.

The interesting systems claim would then be **one expensive world observation,
many isolated typed decisions**, not merely "vision model returns A/B/C."

## Safety of the public API

The HTTP API now defaults to topology 2, the isolated shared-prefix batch, for
multi-question requests.

The packed question-sheet path remains available to benchmarks and experiments,
but is intentionally not the production API default because the current evidence
already shows semantic drift on at least one demo question.
