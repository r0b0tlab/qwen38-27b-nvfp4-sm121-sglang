# Qwen3.8-27B NVFP4 on SGLang (DGX Spark / SM121)

Click-run package for official SGLang day-0 on one GB10. Measured on
`lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6`
with the published 4-of-4 body [`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121).
Production drafter: [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
(`DFlash2DraftModel`, block 8 = K8; overlay image below). Prior candidate:
official [`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)
(native DSPARK K7).

Do **not** use the vLLM `Qwen3DSparkModel` adapter. SGLang wants raw
`DSparkDraftModel` and `--speculative-algorithm DSPARK` (or `DFlash2DraftModel`
+ `--speculative-algorithm DFLASH` for the DFlash2 profile).

Do **not** use `unsloth/Qwen3.8-27B-NVFP4` (SGLang #34895).

Canary: `19 × 23 → 437`. `417` is the FP8-KV flag defect.

## Click-run

```bash
# DFlash2 (NEW production winner — 2026-08-19)
bash scripts/click_run_dflash2.sh

# native DSPARK (prior production candidate)
bash scripts/click_run_dspark.sh

# official EAGLE 3/1/4 (in-checkpoint MTP)
bash scripts/click_run_eagle.sh
```

Each script: download weights if missing, resolve the image, launch, fail-closed
`/health` (90 probes), semantic 10-check. For humans and agents: run the one
command, wait for `ready after N probes`, and expect the semantic gate to print
`"passed": true`. All three default to port 30000 and `--mem-fraction-static 0.70`.

Manual:

```bash
# DFlash2 (production winner)
MODEL_DIR=~/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 \
DRAFT_DIR=~/models/z-lab/Qwen3.8-27B-DFlash2 \
IMAGE=qwen38-27b-sglang-dflash2-sm121:0.2.0 \
  bash scripts/serve.sh dflash2

# DSpark
MODEL_DIR=~/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 \
DRAFT_DIR=~/models/RadixArk/Qwen3.8-27B-DSpark \
  bash scripts/serve.sh dspark
```

### Prerequisites (any machine with one GB10-class GPU + Docker)

1. Docker with NVIDIA Container Toolkit (`docker run --gpus all` works).
2. Python 3 with `huggingface-cli` on PATH for the click-run downloads
   (`pip install -U "huggingface_hub[cli]"`). Anonymous downloads suffice —
   both checkpoints are public.
3. ~25 GB free disk (21 GB body + 4 GB DFlash2 draft) plus Docker image space.
4. Ports 30000 free (override with `PORT=`).

Everything else the scripts do themselves: fetch checkpoints if missing,
validate draft architecture (`DFlash2DraftModel` with conv+selector fields),
build the overlay image from `docker/Dockerfile.dflash2` if absent, launch,
fail-closed `/health` (90 × 10 s probes), and run the 10-check semantic gate.
Expected end state: `ready after N probes` then `"passed": true` and the
canary `19 × 23 → 437`.

Default port **30000**. Default `--mem-fraction-static 0.70` (NIAH used 0.85).
DSpark/EAGLE default to the pinned digest above; DFlash2 defaults to the
overlay tag. Override with `IMAGE=`.

## Container

`docker/Dockerfile.official` is a FROM-pin of the cookbook image, not a
rebuild. Pull by digest:

```bash
docker pull lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6
```

`docker/Dockerfile.dflash2` builds the DFlash2 image from that same digest
plus the seven files of sgl-project/sglang PR #35371 @ `c14312a`
(DFlash2 local convolution + candidate selector, merged 2026-08-19) and a
small r0b0tlab compat layer: an import rename, a ported
`compute_spec_logprobs`, and a quantized-lm_head candidate path for our
W4A16 NVFP4 head (`lm_head.quant_method.apply`, the same kernel
`LogitsProcessor` uses; verify stays lossless through the target's real
logits). Build locally:

```bash
docker build -f docker/Dockerfile.dflash2 -t qwen38-27b-sglang-dflash2-sm121:0.2.0 docker/
```

Nightly / unpinned `lmsysorg/sglang:latest` is not this campaign.

## Profiles (`scripts/serve.sh`)

| Profile | Extra flags | Use |
|---|---|---|
| `ar` | none | correctness baseline |
| `eagle` | EAGLE 3/1/4 | cookbook MTP |
| `dspark` | native DSPARK, block 7, official draft | prior candidate |
| `dflash2` | native DFLASH, block 8, z-lab DFlash2 draft | **production winner** |

Every profile: FlashInfer, `--kv-cache-dtype auto`, `--disable-prefill-cuda-graph`,
`--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, context 262144.

## Think-off results (matched `run_perf_suite.sh`)

Same protocol as the vLLM repo: dedicated c1 random 512→2048 ×5 median;
ladder 1024→256 c1/2/4/8 ×3; ignore-eos; temp 0; seed 0;
`enable_thinking=false`.

| | dedicated c1 med | ladder best c1 / c2 / c4 / c8 | NIAH 8-pt | Quality-200 |
|---|---:|---|---|---|
| SGLang DFlash2 | **28.38** | **23.47 / 36.14 / 54.99 / 92.05** | full-M 3/3 (r0b0bench, below) | flex 82.5 / HE 39/40 / IF 37/40 / ag 16/20 |
| SGLang EAGLE | **25.62** | 27.65 / 43.74 / 74.31 / 123.90 | 8/8 | not run (DSpark Q200 stands) |
| SGLang DSpark | **20.97** | 16.96 / 26.20 / 48.15 / 82.53 | 8/8 | flex 82.5 / HE 39/40 / IF 37/40 / ag 18/20 |
| vLLM MTP K3 | 27.83 | 19.24 / 32.00 / 34.61 / 82.89 | 8/8 | flex 81.25 / HE 39/40 / IF 37/40 / ag 17/20 |
| vLLM DSpark K7 | 28.46 | 16.05 / 28.47 / 43.88 / 61.53 | 8/8 | flex 82.5 / HE 39/40 / IF 37/40 / ag **19/20** |

### DFlash2 vs prior winning profile (SGLang DSpark K7) — side by side

Same node, same image base, same checkpoint, same harness, same run-id
protocol (`dflash2-20260819T014205Z`, zero errors, 17 runs). Accept length is
the mean drafted tokens accepted per verify step.

| Lane (think-off) | DSpark K7 | DFlash2 K8 | Δ |
|---|---:|---:|---:|
| dedicated c1 median | 20.97 | **28.38** | **+35%** |
| ladder c1 | 16.96 | **23.47** | +38% |
| ladder c2 | 26.20 | **36.14** | +38% |
| ladder c4 | 48.15 | **54.99** | +14% |
| ladder c8 | 82.53 | **92.05** | +12% |
| accept length (dedicated c1) | 2.26 | **2.89** | +28% |
| accept length (c8) | — | **3.58** | — |
| Q200 GSM8K flex e2e tok/s (mean) | 37.59 | **52.63** | **+40%** |
| Q200 GSM8K flex e2e tok/s (median) | 37.33 | **52.59** | +41% |
| Q200 GSM8K flex score | 66/80 = 82.5% | **66/80 = 82.5%** | parity |
| Q200 HumanEval | 39/40 | 39/40 | parity |
| Q200 IFEval | 37/40 | 37/40 | parity |
| Q200 agentic (n=20) | 18/20 | 16/20 | −2 (small-n) |
| Q200 all-200 e2e mean | 38.09 | **47.53** | +25% |

Quality is parity (identical GSM8K flex / HumanEval / IFEval; agentic is a
20-item set). Speed is the win: every throughput lane and every per-family
e2e tok/s improves. File: `perf-summary-dflash2.json`,
`quality-200-sglang-dflash2.json`.

DFlash2 quality-200 per-family e2e tok/s (think-off, completion/wall):

| Family | score | mean | median | top |
|---|---|---:|---:|---:|
| GSM8K flex | 66/80 = 82.5% | 52.6 | 52.6 | 64.3 |
| HumanEval | 39/40 | 58.5 | 59.2 | 64.2 |
| IFEval | 37/40 | 23.5 | 22.5 | 49.8 |
| Agentic | 16/20 | 51.5 | 51.0 | 60.8 |
| Hard reasoning | 20 written, not auto-graded | 49.4 | 51.0 | 60.3 |
| All 200 | — | 47.5 | 51.6 | 64.3 |

NIAH is the vLLM `niah-results.json` shape (7726 / 31000 / 124132 + 247738 × 5,
code `QWEN38-NIAH-9X4K`, 512 gen). Files: `niah-results-eagle.json`,
`niah-results-dspark.json`.

Quality set: `artifacts/quality-200.jsonl` sha256 `ca35650e…`.

DSpark quality-200 (think-off, client e2e tok/s = completion_tokens / wall):

| Family | score | mean | median | top |
|---|---|---:|---:|---:|
| GSM8K flex / numeric | 66/80 = 82.5% · 74/80 = 92.5% | 37.6 | 37.3 | 47.0 (`gsm8k-010`) |
| HumanEval | 39/40 | 58.9 | 59.6 | **66.8** (`humaneval-031`) |
| IFEval | 37/40 | 18.2 | 16.4 | 38.4 (`ifeval-032`) |
| Agentic | 18/20 | 43.0 | 43.3 | 50.7 (`agentic-08`) |
| Hard reasoning | 20 written, not auto-graded | 33.5 | 32.9 | 43.5 (`hard-04`) |
| All 200 | — | 38.1 | 37.7 | **66.8** |

Token-weighted overall 32.7 tok/s (77.4k tokens / 39.4 min). File: `quality-200-sglang-dspark.json`.

Matched vLLM DSpark on the same 200: GSM8K / HumanEval / IFEval identical; vLLM agentic 19/20; vLLM e2e mean 33.4 / top 57.5. See the sibling repo `quality-200-vllm-dspark.json`.

Think-on throughput uses the same `scripts/run_perf_suite.sh` with
`EXTRA_BODY_FILE=reference/think-on-extra.json`
(`{"chat_template_kwargs":{"enable_thinking":true}}`).
Quality and NIAH are think-off only.

| Think-on | dedicated c1 med | ladder best c1 / c2 / c4 / c8 |
|---|---:|---|
| SGLang EAGLE | 23.74 | 23.50 / 33.53 / 56.47 / 92.97 |
| SGLang DSpark | 19.83 | 24.28 / 33.09 / 51.59 / 80.94 |
| vLLM MTP K3 | 29.12 | 22.57 / 37.46 / 60.22 / 84.07 |
| vLLM DSpark K7 | 28.65 | 16.13 / 28.63 / 44.63 / 62.23 |

Sibling vLLM package: [`r0b0tlab/qwen38-27b-nvfp4-sm121-vllm`](https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-vllm).

## r0b0bench core-subset (DFlash2, 2026-08-19)

Full 11-lane `core-subset` (r0b0bench `1.0.0rc2` @ `ae4e099`) against the
DFlash2 profile on one GB10, thinking off. **11/11 lanes PASS,
`infra_errors_total=0`, `invalid_for_publish=false`.** report.json sha256
`e2c2d50defe1dde9…`.

Draft-block (K) sweep first — one variable (`DFLASH_BLOCK_SIZE`), canary
`19 × 23 → 437` on every row, accept = SGLang mean accept-length gauge:

| block | c1 median tok/s | c8 best agg tok/s | accept length |
|---:|---:|---:|---:|
| 6 | 25.28 | 158.23 | 3.35 |
| 7 | 25.79 | 157.82 | 3.60 |
| **8** | **27.88** | 153.90 | 3.58 |
| 9 | 26.05 | 123.16 | 3.59 |

Block 8 wins c1 by ~8–10% over 6/7; block 9 collapses c8 (−22%). K* = 8
(the draft's native `block_size`). All core-subset rows below are block 8.

| Lane | Result |
|---|---|
| canary | PASS |
| BFCL-MT (official `multi_turn_base`, 200) | **0.69** |
| BFCL-AST-600 (multiple / parallel / parallel_multiple) | µ **0.2733** (0.18 / 0.46 / 0.18) |
| latency (c1 streaming) | TTFT **214.6 ms** mean |
| concurrency (512-out, portable) | C1 68.6 / C2 124.3 / C4 212.0 / C6 **276.4** agg tok/s |
| throughput | decode **26.02** tok/s median (2048-out ×5) · prefill **22,663** server tok/s (~22.8k-token prompts) |
| NIAH (full max-context) | 65,472 / 130,944 / **235,699** all PASS, exact retrieval (90% in 336 s) |
| QA (ARC-Easy MC) | **0.9625** @400 |
| IFEval (lightweight constraint scorer) | **0.82** @200 |
| HumanEval | pass@1 **0.8720** @164 |
| GSM8K (0-shot flexible-extract) | **0.865** @200 |

Supplementary extended ladder (diagnostic, portable backend, 512-out,
think-off, zero errors): c1 29.3 / c2 49.6 / c4 84.2 / c8 131.1 /
**c16 178.9** / c32 158.6 agg tok/s — saturation peak at c16
(`max_running_requests=48`).

AST misses are model-side, not harness: the official adapter decoded
≥197/200 rows to structured calls in every category; the misses are strict
AST function-choice/argument mismatches plus ≤3 prose-fallback rows.

Serve deltas for this run only: `--enable-metrics` (telemetry) and
`--mem-fraction-static 0.73` — a 0.70 boot drew 234,240 KV tokens, below
the 235,955 the 90% NIAH probe needs; 0.73 yields 262,611 ≥ the full
262,144 window. Results-ledger entry:
`qwen38-27b-nvfp4-sglang-dflash2-k8-core-subset-20260819`.

Telemetry (15 s sampling across the whole campaign): suite window GPU util
mean 95.7%, power mean 45.4 W (max 88.4 W), temp max 88 °C, spec
accept-length mean 4.63.

## llama-benchy (diagnostic, not the claim row)

`scripts/run_llama_benchy.sh` needs `--extra-body return_token_ids=false`
or SGLang 400s the stream. Claim numbers come from `run_perf_suite.sh`.

## License

MIT for scripts/docs. Weights follow Qwen / RadixArk / r0b0tlab cards and
are not stored in this git tree.
