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
- dflash2
- dspark
- eagle
- speculative-decoding
---

# Qwen3.8-27B on SGLang (NVFP4 + native DFlash2 / DSpark / EAGLE)

Serve recipe for the published 4-of-4 body
[`r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121)
on official SGLang. Production draft: DFlash2
[`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
(`DFlash2DraftModel`, block 8 = K8). Prior candidate draft for DSpark:
[`RadixArk/Qwen3.8-27B-DSpark`](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)
(`DSparkDraftModel`, not the vLLM adapter).

Image pin: `lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6`

```bash
bash scripts/click_run_dflash2.sh   # production winner (DFlash2 K8, z-lab draft)
# or
bash scripts/click_run_dspark.sh    # prior candidate (native DSPARK K7)
# or
bash scripts/click_run_eagle.sh     # in-checkpoint MTP (EAGLE 3/1/4)
```

Think-off matched `run_perf_suite.sh` (512→2048 dedicated c1 / 1024→256 ladder):

| | c1 med | c8 best | NIAH |
|---|---:|---:|---|
| DFlash2 K8 (z-lab draft) | **28.38** | 92.05 | full-M 3/3 (r0b0bench) |
| EAGLE 3/1/4 | 25.62 | 123.90 | 8/8 |
| native DSPARK K7 | 20.97 | 82.53 | 8/8 |

Think-on (same suite): EAGLE c1 **23.74** / c8 92.97; DSpark c1 **19.83** / c8 80.94.

DFlash2 quality-200 think-off: GSM8K flex 82.5% / HE 39/40 / IF 37/40 / ag 16/20;
e2e mean 47.5 / top 64.3 tok/s — quality parity with DSpark at +25–40% e2e speed.

r0b0bench core-subset (DFlash2, think-off, 2026-08-19): 11/11 lanes PASS,
zero infra errors — GSM8K 0.865, ARC-Easy 0.9625, IFEval 0.82 (lightweight),
HumanEval 0.872, BFCL-MT 0.69, full-262144-context NIAH 3/3
(65,472/130,944/235,699), decode 26.0 tok/s, prefill 22.7k tok/s.

Canary `19×23→437`. Full tables live in the GitHub repo
`r0b0tlab/qwen38-27b-nvfp4-sm121-sglang`.
