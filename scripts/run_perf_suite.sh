#!/usr/bin/env bash
# Match vLLM repo scripts/run_perf_suite.sh:
#   dedicated c1: random 512→2048, 1 prompt, c=1, ignore-eos, temp 0, seed 0, 5 reps, median
#   ladder: random 1024→256, levels 1 2 4 8, 3 reps, prompts=max(8, c*4)
# extra-request-body is passed via a file (docker argv mangles JSON braces).
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-qwen38-27b}"
CONTAINER="${CONTAINER:-qwen38-sglang}"
MODEL_PATH="${MODEL_PATH:-/model}"
RUN_ID="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-$ROOT/evidence/perf/$RUN_ID}"
mkdir -p "$OUT"

LEVELS="${LEVELS:-1 2 4 8}"
REPS="${REPS:-3}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
C1_TOKENS="${C1_TOKENS:-2048}"
EXTRA_BODY_FILE="${EXTRA_BODY_FILE:-}"
if [[ -n "${EXTRA_BODY_FILE}" ]]; then
  EXTRA_JSON="$(cat "$EXTRA_BODY_FILE")"
else
  EXTRA_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
fi
python3 -c 'import json,sys; json.loads(sys.argv[1])' "$EXTRA_JSON"
printf '%s' "$EXTRA_JSON" > "$OUT/extra_request_body.json"
docker cp "$OUT/extra_request_body.json" "$CONTAINER:/tmp/extra_request_body.json"
docker cp "$ROOT/scripts/bench_one.py" "$CONTAINER:/tmp/bench_one.py" 2>/dev/null || docker cp "$ROOT/bench_one.py" "$CONTAINER:/tmp/bench_one.py"

{
  echo "run_id=$RUN_ID"
  echo "container=$CONTAINER port=$PORT model=$MODEL"
  echo "levels=$LEVELS reps=$REPS input=$INPUT_LEN output=$OUTPUT_LEN c1_tokens=$C1_TOKENS"
  echo "protocol=vllm-repo-run_perf_suite.sh"
  echo "extra_request_body=$EXTRA_JSON"
  date -u -Is
  curl -fsS "http://127.0.0.1:$PORT/v1/models"
} | tee "$OUT/preflight.txt"

bench () {
  local fname="$1"; shift
  local remote="/tmp/${RUN_ID}-${fname}"
  docker exec "$CONTAINER" rm -f "$remote"
  docker exec "$CONTAINER" python3 /tmp/bench_one.py "$remote" "$@"
  docker cp "$CONTAINER:${remote}" "$OUT/${fname}" >/dev/null
  python3 - "$OUT/$fname" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
ot=d.get("output_throughput")
if ot is None:
    ot=(d.get("total_output_tokens") or d.get("output_len") or 0)/max(d.get("duration") or 1,1e-9)
print(f"  {sys.argv[1].split('/')[-1]}: output_throughput={ot:.2f} tok/s completed={d.get('completed')} accept={d.get('accept_length')} errors={d.get('errored') or d.get('error_n')}")
PY
}

echo "=== dedicated c1 (max_tokens=$C1_TOKENS, 5 reps) ===" | tee -a "$OUT/progress.log"
for rep in 1 2 3 4 5; do
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null || { echo "SERVER_DOWN c1-r${rep}"; exit 2; }
  bench "qwen38-dedicated-c1-r${rep}.json" \
    --random-input-len 512 --random-output-len "$C1_TOKENS" \
    --num-prompts 1 --max-concurrency 1 \
    >"$OUT/dedicated-c1-r${rep}.log" 2>&1 || { echo "C1_R${rep}_FAIL" | tee -a "$OUT/progress.log"; exit 2; }
done

for c in $LEVELS; do
  prompts=$(( c * 4 )); (( prompts < 8 )) && prompts=8
  for rep in $(seq 1 "$REPS"); do
    echo "=== c=${c} rep=${rep} prompts=${prompts} $(date -u -Is) ===" | tee -a "$OUT/progress.log"
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null || { echo "SERVER_DOWN c${c}-r${rep}"; exit 2; }
    bench "qwen38-c${c}-r${rep}.json" \
      --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
      --num-prompts "$prompts" --max-concurrency "$c" \
      >"$OUT/c${c}-r${rep}.log" 2>&1 || { echo "C${c}_R${rep}_FAIL" | tee -a "$OUT/progress.log"; exit 2; }
  done
done
echo "SUITE_DONE $RUN_ID" | tee -a "$OUT/progress.log"
