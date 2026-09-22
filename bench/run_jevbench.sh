#!/usr/bin/env bash
set -euo pipefail

endpoint="${1:-http://127.0.0.1:8080}"
work="$(mktemp -d "${TMPDIR:-/tmp}/gemma-jevbench.XXXXXX")"
harness="$work/jevbench"
tasks="$harness/datasets/public/easy.jsonl,$harness/datasets/public/original.jsonl,$harness/datasets/public/hard.jsonl"

git clone --quiet --depth 1 --branch v1.3.0 \
  https://github.com/fstandhartinger/jevbench "$harness"

PYTHONPATH="$harness" python3 -m jevbench.cli run \
  --tasks "$tasks" \
  --adapter typesafe \
  --endpoint "$endpoint" \
  --model gemma-system-one-4b-q4 \
  --key-env '' \
  --results "$work/results.jsonl" \
  --ledger "$work/ledger.jsonl" \
  --raw-dir "$work/raw" \
  --cap-usd 0 \
  --reserve-usd 0 \
  --cost-basis local_no_provider_tariff \
  --manifest "$work/manifest.json" \
  --run-label gemma-system-one-public-v1.3

PYTHONPATH="$harness" python3 -m jevbench.cli summarize \
  --tasks "$tasks" \
  --results "$work/results.jsonl" \
  --ledger "$work/ledger.jsonl" \
  | tee "$work/summary.json"

printf '\nArtifacts: %s\n' "$work"
