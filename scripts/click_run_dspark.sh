#!/usr/bin/env bash
# Click-run native DSPARK: 4-of-4 NVFP4 body + official DSpark draft + pinned image + canary.
set -euo pipefail
CKPT="${CKPT:-$HOME/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121}"
DRAFT="${DRAFT:-$HOME/models/RadixArk/Qwen3.8-27B-DSpark}"
IMAGE="${IMAGE:-lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6}"
MEM_FRACTION="${MEM_FRACTION:-0.70}"
mkdir -p "$(dirname "$CKPT")" "$(dirname "$DRAFT")"
if [ ! -f "$CKPT/model-00001-of-00004.safetensors" ]; then
  huggingface-cli download r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 --local-dir "$CKPT"
fi
for shard in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  [ -f "$CKPT/$shard" ] || { echo "still missing $shard after download"; exit 2; }
done
if [ ! -f "$DRAFT/model.safetensors" ] && [ ! -f "$DRAFT/model-00001-of-00001.safetensors" ]; then
  huggingface-cli download RadixArk/Qwen3.8-27B-DSpark --local-dir "$DRAFT"
fi
python3 - "$DRAFT/config.json" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
if cfg.get("architectures") != ["DSparkDraftModel"]:
    raise SystemExit("draft is not official DSparkDraftModel; do not use the vLLM Qwen3DSparkModel adapter")
PY
docker pull "$IMAGE"
MODEL_DIR="$CKPT" DRAFT_DIR="$DRAFT" IMAGE="$IMAGE" MEM_FRACTION="$MEM_FRACTION" \
  bash "$(dirname "$0")/serve.sh" dspark
ready=0
for i in $(seq 1 90); do
  if curl -fsS -m 5 http://127.0.0.1:30000/health >/dev/null 2>&1; then
    echo "ready after ${i} probes"
    ready=1
    break
  fi
  if ! docker inspect -f '{{.State.Running}}' qwen38-sglang 2>/dev/null | grep -q true; then
    echo "container died during startup; last logs:" >&2
    docker logs --tail 40 qwen38-sglang >&2 || true
    exit 3
  fi
  sleep 10
done
if [ "$ready" != 1 ]; then
  echo "READY_TIMEOUT after 90 probes" >&2
  docker logs --tail 40 qwen38-sglang >&2 || true
  exit 4
fi
python3 "$(dirname "$0")/run_semantic_gate.py" --base-url http://127.0.0.1:30000 --output /tmp/qwen38-sglang-semantic.json
python3 - <<'PY'
import json
d=json.load(open("/tmp/qwen38-sglang-semantic.json"))
print(json.dumps({"passed": d.get("passed"), "checks": d.get("checks")}, indent=2))
raise SystemExit(0 if d.get("passed") else 1)
PY
