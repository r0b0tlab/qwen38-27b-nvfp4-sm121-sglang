---
license: mit
language: en
pipeline_tag: text-generation
tags:
- nvfp4
- sglang
- sm121
- gb10
- dgx-spark
- dspark
- eagle
- speculative-decoding
---

# Qwen3.8-27B on SGLang (NVFP4 + native DSpark / EAGLE)

Serve recipe for the published 4-of-4 body
[`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121)
on official SGLang. Draft for DSpark:
[`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)
(`DSparkDraftModel`, not the vLLM adapter).

Image pin: `lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6`

```bash
bash scripts/click_run_dspark.sh
# or
bash scripts/click_run_eagle.sh
```

Think-off matched `run_perf_suite.sh` (512→2048 dedicated c1 / 1024→256 ladder):

| | c1 med | c8 best | NIAH |
|---|---:|---:|---|
| EAGLE 3/1/4 | 25.62 | 123.90 | 8/8 |
| native DSPARK | 20.97 | 82.53 | 8/8 |

Think-on (same suite): EAGLE c1 **23.74** / c8 92.97; DSpark c1 **19.83** / c8 80.94.

DSpark quality-200 think-off: GSM8K flex 82.5% / HE 39/40 / IF 37/40 / ag 18/20; e2e mean 38.1 / top **66.8**.

Canary `19×23→437`. Full tables live in the GitHub repo
`r0b0tlab/qwen38-27b-nvfp4-sm121-sglang`.
