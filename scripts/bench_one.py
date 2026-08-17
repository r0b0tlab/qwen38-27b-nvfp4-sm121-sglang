#!/usr/bin/env python3
import json
import subprocess
import sys

out = sys.argv[1]
rest = sys.argv[2:]
body = open("/tmp/extra_request_body.json").read()
json.loads(body)
cmd = [
    "python3",
    "-m",
    "sglang.bench_serving",
    "--backend",
    "sglang-oai-chat",
    "--base-url",
    "http://127.0.0.1:30000",
    "--model",
    "qwen38-27b",
    "--tokenizer",
    "/model",
    "--dataset-name",
    "random",
    "--seed",
    "0",
    "--temperature",
    "0",
    "--request-rate",
    "inf",
    "--extra-request-body",
    body,
    "--output-file",
    out,
    "--flush-cache",
] + rest
print("EXTRA_BODY", body, flush=True)
raise SystemExit(subprocess.call(cmd))
