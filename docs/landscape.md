# Decision-model landscape

This note maps the projects around Jev/System-One-style inference by **what they
actually contribute**, not by whether they use the word "Jev".

The space is easier to understand as four layers:

1. **Purpose-trained decision models** — train the representation/readout for
   bounded decisions rather than reusing a chat model unchanged.
2. **Inference-time decision adapters** — reuse a normal LM/VLM and read a
   restricted distribution instead of generating prose.
3. **Calibration and debiasing** — make the returned distribution useful as a
   probability rather than merely a model-relative score.
4. **Shared-state execution systems** — amortize one expensive state across many
   independent questions without changing their semantics.

## Purpose-trained decision models

| Project | Main idea | What it offers | Relation to this repo |
| --- | --- | --- | --- |
| [TypeSafe Jev](https://typesafe.ai/) | Purpose-built System One model trained for bounded decisions | `choice`, `noul`, `score`, calibrated hosted API, parallel evaluation | The interface and problem definition this repo studies; not an open model clone |
| [Laya](https://github.com/he-jev/laya) | Small encoder-style non-autoregressive decision models | Open weights, low parameter count, multilingual decision inference | Shows that a causal chat LM is not required |
| [Kev](https://github.com/jaredpalmer/kev) | Qwen-based trained decision models with explicit question isolation and decision heads | Open training/inference, several sizes, TypeSafe-shaped API, isolation-aware batching | Closest reference for the semantics-preserving multi-question problem |
| [Mapika Decider](https://github.com/Mapika/decider) | Family of trained decision models, including outcome/calibration-oriented training and vision variants | Multiple model sizes, multimodal work, agent/browser/game decision tasks | Useful reference for training and multimodality |
| [system-one-open](https://github.com/mithalouni/system-one-open) | Train Gemma variants directly for one-pass typed decisions | Open Gemma-based decision-model experiment | Natural "train the readout/model" continuation of this repo |

## Inference-time adapters over ordinary models

| Project | Main idea | What it offers | Relation to this repo |
| --- | --- | --- | --- |
| [OpenJev](https://github.com/lookski/openjev) | Restrict ordinary LM logits to legal choices | Very small open baseline for Jev-shaped decisions | Establishes that the basic restricted-logit trick is commodity |
| [SemIf](https://github.com/TheoLeeCJ/SemIf) | Multi-backend semantic decision inference with shared-state modes | Qwen/llama.cpp/MLX-style portability and prefix reuse | Directly adjacent on shared-state inference |
| [openjev-sglang](https://github.com/ekzhang/openjev-sglang) | Run a capable open model behind an optimized SGLang serving stack | Prefix/radix caching and production-style serving | Shows that serving quality can matter as much as a new model |
| [djev](https://github.com/mmastrac/djev) | Use DiffusionGemma / diffusion answer slots rather than ordinary autoregressive generation | Naturally parallel answer positions | Alternative architecture for many simultaneous decisions |
| Allan R. B. Olesen's Jev-like wrapper | Force bounded answers through local/API models and read logprobs; includes attachments | Very portable, multimodal wrapper pattern | Strong on portability/multimodality; weaker control over exact logits/cache semantics |
| This repo | Own the Gemma/Candle hot path and gather exact legal-label logits | Native Rust, exact logits, shared-state KV reuse, batched suffixes, experimental tiny-pointer sheet | Focus is moving from "can Gemma imitate Jev?" to semantic isolation under optimization |

## Calibration / debiasing

| Project | Main idea | Why it matters |
| --- | --- | --- |
| [AnyJev](https://github.com/nokia-applied-research/AnyJev) | Correct option-order / label-prior bias, fit calibration, and explore intermediate-layer decision readouts | Raw restricted softmax is not automatically a calibrated probability |
| Kev / Decider / Jev | Train or fit the model/readout with calibration as an explicit objective | Better suited to autonomous threshold decisions than raw LM logits |

This repo currently reports raw restricted softmax and explicitly labels it
**uncalibrated**. JevBench's public run measured substantial overconfidence, so
calibration is a future research axis rather than a claim.

## Shared-state execution

The systems problem predates Jev:

- [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
  reuses exact KV prefixes.
- [Hydragen](https://arxiv.org/abs/2402.05099) separates shared-prefix attention
  from unique suffix attention.
- [DeFT](https://arxiv.org/abs/2404.00242) generalizes efficient attention to
  tree-structured shared-prefix workloads.
- Kev explicitly isolates question branches while reusing state computation.

The key distinction for this repo is between:

```text
SAFE/REFERENCE SHAPE

state prefix
    |
    +---- question A ---- decision A
    +---- question B ---- decision B
    +---- question C ---- decision C

Each branch computes P(y | state, its_question).
```

and:

```text
PACKED QUESTION-SHEET SHAPE

state
question A
question B
question C
    |
tiny pointer "answer #B"
    |
decision B

This computes P(y | state, A, B, C, pointer), which is not the same function.
```

The current repository has already observed a concrete answer change between
those topologies. That observation motivates the semantic-isolation benchmark.

## What is commodity now

These are no longer sufficient claims on their own:

- "No generated JSON."
- "Read A/B/C logits."
- "Return a typed choice."
- "Use a local open model."
- "Expose a Jev-compatible endpoint."

They are useful engineering, but many independent projects now do them.

## Open territory

The stronger combined problem is:

```text
runtime-defined questions
+ one expensive shared state
+ exact/cheap many-question execution
+ question isolation
+ calibrated probabilities
+ long-context robustness
+ multimodality
```

This repo is targeting the **execution + isolation** slice first.

See [semantic-isolation.md](semantic-isolation.md) for the concrete benchmark and
implementation plan.
