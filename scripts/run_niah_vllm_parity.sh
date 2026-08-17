#!/usr/bin/env bash
# vLLM-parity NIAH: same 8 points as
#   projects/qwen38-27b-nvfp4-sm121-vllm/repo/niah-results.json
#   README.md "NIAH 8/8 @ 262,144"
#   CAMPAIGN-CERTIFICATION.json niah_262144
#   scripts/run_niah.py --code QWEN38-NIAH-9X4K
#
# Ladder: 7726 / 31000 / 124132 at needle 0.5
# Long:   247738 at needle 0.05 0.25 0.50 0.75 0.95
# Gen:    512 (certification: "512 gen past retrieval")
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
NIAH="${NIAH:-$ROOT/scripts/run_niah.py}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000/v1}"
TOKENIZER="${TOKENIZER:?set TOKENIZER to the served checkpoint}"
OUT="${OUT:?set OUT to niah-results.json}"
WORKDIR="$(dirname "$OUT")"
mkdir -p "$WORKDIR"

run_one() {
  local tag="$1" depths="$2" frac="$3"
  $PY "$NIAH" \
    --base-url "$BASE_URL" \
    --model qwen38-27b \
    --tokenizer "$TOKENIZER" \
    --depths $depths \
    --needle-fraction "$frac" \
    --max-tokens 512 \
    --code QWEN38-NIAH-9X4K \
    --protocol-label "vllm-parity" \
    --output "$WORKDIR/_part-${tag}.json"
}

run_one ladder-8k32k131k "7726 31000 124132" 0.5
run_one long-p05 247738 0.05
run_one long-p25 247738 0.25
run_one long-p50 247738 0.50
run_one long-p75 247738 0.75
run_one long-p95 247738 0.95

$PY - "$WORKDIR" "$OUT" <<'PY'
import json,sys
from pathlib import Path
wd, out = Path(sys.argv[1]), Path(sys.argv[2])
order = [
    ("ladder-8k32k131k", None),
    ("long-p05", 0.05),
    ("long-p25", 0.25),
    ("long-p50", 0.50),
    ("long-p75", 0.75),
    ("long-p95", 0.95),
]
rows=[]
for tag, frac in order:
    p=wd/f"_part-{tag}.json"
    d=json.loads(p.read_text())
    for r in d.get("rows") or []:
        rows.append({
            "depth": frac if frac is not None else 0.5,
            "prompt_tokens": r.get("prompt_tokens_constructed"),
            "pass": bool(r.get("passed")),
            "elapsed_s": round(float(r.get("elapsed_s") or 0), 1),
            "answer": (r.get("response_text") or "").strip(),
            "http_status": r.get("http_status"),
            "needle_frac": r.get("needle_fraction_effective"),
            "error": r.get("error"),
        })
out.write_text(json.dumps(rows, indent=2)+"\n")
ok=sum(1 for r in rows if r["pass"])
print(json.dumps({"output":str(out),"pass_count":ok,"total":len(rows),"all_passed":ok==len(rows) and len(rows)==8}, indent=2))
PY
