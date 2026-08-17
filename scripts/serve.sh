#!/usr/bin/env bash
# Official SGLang day-0 launch for Qwen3.8-27B NVFP4 on one GB10.
# Usage: MODEL_DIR=... [DRAFT_DIR=...] [IMAGE=...] bash scripts/serve.sh {ar|eagle|dspark}
set -euo pipefail
PROFILE="${1:-dspark}"
IMAGE="${IMAGE:-lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the NVFP4 checkpoint}"
DRAFT_DIR="${DRAFT_DIR:-}"
NAME="${NAME:-qwen38-sglang}"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.70}"

COMMON=(
  sglang
  serve
  --trust-remote-code
  --model-path /model
  --served-model-name qwen38-27b
  --host 0.0.0.0
  --port 30000
  --attention-backend flashinfer
  --kv-cache-dtype auto
  --chunked-prefill-size 8192
  --max-prefill-tokens 8192
  --context-length 262144
  --mem-fraction-static "$MEM_FRACTION"
  --disable-prefill-cuda-graph
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --mamba-full-memory-ratio 4.59
)

VOLS=(-v "$MODEL_DIR:/model:ro")
case "$PROFILE" in
  ar) EXTRA=() ;;
  eagle)
    EXTRA=(
      --speculative-algorithm EAGLE
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
    )
    ;;
  dspark)
    [ -n "$DRAFT_DIR" ] && [ -f "$DRAFT_DIR/config.json" ] || { echo "DRAFT_DIR required for dspark"; exit 4; }
    VOLS+=(-v "$DRAFT_DIR:/draft:ro")
    EXTRA=(
      --speculative-algorithm DSPARK
      --speculative-draft-model-path /draft
      --speculative-dspark-block-size 7
      --speculative-draft-model-quantization unquant
    )
    ;;
  *) echo "profile ar|eagle|dspark"; exit 6 ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all \
  --cpus 14 --memory 100g --memory-swap 100g \
  --shm-size 32g --ipc=host \
  -p "$PORT:30000" \
  "${VOLS[@]}" \
  "$IMAGE" \
  "${COMMON[@]}" \
  "${EXTRA[@]}"
echo "launched $NAME profile=$PROFILE image=$IMAGE port=$PORT mem=$MEM_FRACTION"
