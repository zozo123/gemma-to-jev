# Gemma System One

Turn a small Gemma into a language-conditioned decision function, in native
Rust. This demo uses `google/gemma-3-4b-it` (Q4_K_M) with Candle on Apple Metal.

Project page: <https://zozo123.github.io/gemma-to-jev/>

Instead of:

```text
prompt -> generate tokens -> parse text
```

it does:

```text
prompt -> one forward pass -> legal-label logits -> softmax -> typed answer
```

The decision path has no autoregressive decoding, no JSON, and no parser.

## Run

```bash
cargo run --release
```

The first run downloads about 2.3 GB of weights and the tokenizer from Hugging
Face. Later runs use the local Hugging Face cache.

```bash
cargo run --release -- --repeat 5      # latency percentiles
cargo run --release -- --stress        # determinism, order stability, spot-check
cargo run --release -- --baseline      # compare against real generation
cargo run --release -- --temperature 1.5
cargo run --release -- --cpu
```

Temperature is an inference control, not calibration.

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

## Why this is Jev-like

A conventional LLM spends inference compute producing an answer token by token.
This demo instead asks Gemma for a discriminative decision and terminates after
one forward pass. Natural-language state, questions, and labels stay flexible,
while the output space is constrained at runtime. This is a primitive
System-One-style layer inspired by the same architectural idea, not an
implementation or reproduction of TypeSafe Jev.

## Measured on an Apple M1 Pro

Warm, 4-bit 4B model, `--repeat 5 --stress --baseline`:

| Metric | Result |
| --- | --- |
| Per decision | p50 401 ms · p90 414 ms · p99 419 ms |
| Five-question evaluation | p50 2020 ms |
| Model load (warm page cache) | 5.9 – 8.4 s |
| Determinism | 5/5 identical distributions |
| Option-order stability | 7/7 rotations kept the same decision |
| Labelled failure-class spot-check | 4/4 |
| System One vs `generate()` | 408 ms / 0 tokens vs 804 ms / 2 tokens |

The generation baseline emits only two tokens, so the gap is modest and varies
between runs (477–804 ms observed). The saving grows with longer outputs; the
structural win is that there is no text to parse and the output space cannot go
out of range.

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
- `evaluate` runs questions sequentially. Candle's quantized Gemma exposes no
  cache reset and no ragged attention mask, so fused batching needs upstream
  support.

## Next

- fuse question batches with a correct padding attention mask
- reuse shared-state KV prefill across questions
- calibrate on held-out decisions, then measure ECE and Brier score
- fine-tune on decision data instead of prompting a general chat model
