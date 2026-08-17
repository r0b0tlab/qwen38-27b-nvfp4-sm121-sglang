#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000/v1}"
MODEL="${MODEL:-qwen38-27b}"
TOKENIZER="${TOKENIZER:?}"
OUT="${OUT:?}"
mkdir -p "$OUT"
EXTRA=(--extra-body return_token_ids=false --extra-body 'chat_template_kwargs={"enable_thinking":false}')

uvx llama-benchy \
  --base-url "$BASE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --pp 2048 --tg 128 --depth 0 4096 8192 16384 \
  --runs 3 --latency-mode api --concurrency 1 \
  "${EXTRA[@]}" \
  --save-result "$OUT/llama-benchy-c1-depths.md" --format md \
  | tee "$OUT/llama-benchy-c1-depths.log"

for c in 1 2 4 8; do
  echo "=== CONC $c $(date -u -Is) ==="
  uvx llama-benchy \
    --base-url "$BASE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --pp 2048 --tg 128 --depth 0 \
    --runs 3 --latency-mode api --concurrency "$c" \
    "${EXTRA[@]}" \
    --save-result "$OUT/llama-benchy-c${c}.md" --format md \
    | tee "$OUT/llama-benchy-c${c}.log"
done
echo SUITE_DONE
