#!/usr/bin/env python3
"""19x23 canary against a live OpenAI-compatible endpoint."""
from __future__ import annotations
import argparse, json, re, urllib.request

def chat(base: str, model: str, prompt: str, max_tokens: int = 256) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    body = chat(args.base_url, args.model, "Compute 19 times 23. End your answer with the integer result.", 1024)
    text = (body["choices"][0]["message"].get("content") or "")
    ok = bool(re.search(r"\b437\b", text))
    out = {"passed": ok, "text": text, "usage": body.get("usage")}
    open(args.output, "w").write(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"passed": ok, "has_437": ok, "preview": text[:200]}, indent=2))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
