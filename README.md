# Qwen3.8-27B NVFP4 on SGLang (DGX Spark / SM121)

Click-run package for official SGLang day-0 on one GB10. Measured on
`lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6`
with the published 4-of-4 body [`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121)
and official [`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark).

Do **not** use the vLLM `Qwen3DSparkModel` adapter. SGLang wants raw
`DSparkDraftModel` and `--speculative-algorithm DSPARK`.

Do **not** use `unsloth/Qwen3.8-27B-NVFP4` (SGLang #34895).

Canary: `19 × 23 → 437`. `417` is the FP8-KV flag defect.

## Click-run

```bash
# native DSPARK (production candidate)
bash scripts/click_run_dspark.sh

# official EAGLE 3/1/4 (in-checkpoint MTP)
bash scripts/click_run_eagle.sh
```

Each script: download 4-of-4 shards if missing, `docker pull` the pinned
digest, launch, fail-closed `/health` (90 probes), semantic 10-check.

Manual:

```bash
MODEL_DIR=~/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121 \
DRAFT_DIR=~/models/RadixArk/Qwen3.8-27B-DSpark \
  bash scripts/serve.sh dspark
```

Default port **30000**. Default `--mem-fraction-static 0.70` (NIAH used 0.85).
Image default is the digest above. Override with `IMAGE=`.

## Container

`docker/Dockerfile.official` is a FROM-pin of the cookbook image, not a
rebuild. Pull by digest:

```bash
docker pull lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6
```

Nightly / unpinned `lmsysorg/sglang:latest` is not this campaign.

## Profiles (`scripts/serve.sh`)

| Profile | Extra flags | Use |
|---|---|---|
| `ar` | none | correctness baseline |
| `eagle` | EAGLE 3/1/4 | cookbook MTP |
| `dspark` | native DSPARK, block 7, official draft | production candidate |

Every profile: FlashInfer, `--kv-cache-dtype auto`, `--disable-prefill-cuda-graph`,
`--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, context 262144.

## Think-off results (matched `run_perf_suite.sh`)

Same protocol as the vLLM repo: dedicated c1 random 512→2048 ×5 median;
ladder 1024→256 c1/2/4/8 ×3; ignore-eos; temp 0; seed 0;
`enable_thinking=false`.

| | dedicated c1 med | ladder best c1 / c2 / c4 / c8 | NIAH 8-pt | Quality-200 |
|---|---:|---|---|---|
| SGLang EAGLE | **25.62** | 27.65 / 43.74 / 74.31 / **123.90** | 8/8 | not run (DSpark Q200 stands) |
| SGLang DSpark | **20.97** | 16.96 / 26.20 / 48.15 / 82.53 | 8/8 | flex 82.5 / HE 39/40 / IF 37/40 / ag 18/20 |
| vLLM MTP K3 | 27.83 | 19.24 / 32.00 / 34.61 / 82.89 | 8/8 | flex 81.25 / HE 39/40 / IF 37/40 / ag 17/20 |
| vLLM DSpark K7 | 28.46 | 16.05 / 28.47 / 43.88 / 61.53 | 8/8 | think-off prose |

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

## llama-benchy (diagnostic, not the claim row)

`scripts/run_llama_benchy.sh` needs `--extra-body return_token_ids=false`
or SGLang 400s the stream. Claim numbers come from `run_perf_suite.sh`.

## License

MIT for scripts/docs. Weights follow Qwen / RadixArk / r0b0tlab cards and
are not stored in this git tree.
