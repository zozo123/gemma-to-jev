# Gemma System One

Turn a small Gemma into a language-conditioned decision function. This native
Rust demo uses `google/gemma-3-1b-it` (Q4_K_M) with Candle on Apple Metal. The
270M model was tested first but failed the semantic sanity checks; 1B is the
smallest tested size that produced a credible demo.

Instead of:

```text
prompt -> generate tokens -> parse text
```

it does:

```text
prompt -> one forward pass -> legal-choice logits -> softmax
```

The decision path has no autoregressive decoding, JSON, or parser.

## Run

```bash
cargo run --release
```

The first run downloads about 800 MB of weights and the tokenizer from Hugging
Face. Later runs use the local Hugging Face cache.

```bash
cargo run --release -- --repeat 10
cargo run --release -- --temperature 1.5
cargo run --release -- --cpu
```

Temperature is an inference control, not calibration.

## How it works

```text
                  GEMMA 3 1B

state + question + runtime choices
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

The tokenizer is checked at runtime: each answer label must add exactly one
token at the answer position. Softmax is applied only across those legal token
logits.

## Why this is Jev-like

A conventional LLM spends inference compute producing an answer token by token.
This demo instead asks Gemma for a discriminative decision and terminates after
one forward pass. Natural-language state, questions, and choices stay flexible,
while the output space is constrained at runtime. This is a primitive
System-One-style layer inspired by the same architectural idea, not an
implementation or reproduction of TypeSafe Jev.

## Limitations

- Raw softmax values are model-relative scores, not calibrated probabilities.
- This is a 4-bit quantized 1B model, optimized for local speed over maximum
  decision quality.
- `decide_batch` currently preserves exact unpadded prompts by evaluating rows
  sequentially; a fused padded batch needs attention-mask support in Candle's
  quantized Gemma path.
- Production Jev-like behavior needs a calibration set, temperature scaling,
  Brier loss/ECE measurement, and decision-specific fine-tuning.

## Next

- fuse question batches with a correct padding attention mask
- reuse shared-state KV prefill
- calibrate on held-out decisions
- distill from a larger reasoning model
