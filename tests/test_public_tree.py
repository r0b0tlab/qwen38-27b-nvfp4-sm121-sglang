#!/usr/bin/env python3
"""Fail-closed public-tree checks for the SGLang repro package."""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("192.168.", "r0b0tdgx@", "/home/r0b0tdgx", "HF_TOKEN=", "ghp_")
REQUIRED = (
    "LICENSE",
    "README.md",
    "HF-CARD.md",
    "CAMPAIGN-CERTIFICATION.json",
    "docker/Dockerfile.official",
    "scripts/serve.sh",
    "scripts/click_run_dspark.sh",
    "scripts/click_run_eagle.sh",
    "scripts/click_run_dflash2.sh",
    "scripts/run_semantic_gate.py",
    "scripts/run_quality_set.py",
    "scripts/run_perf_suite.sh",
    "scripts/run_llama_benchy.sh",
    "artifacts/quality-200.jsonl",
    "artifacts/quality-200.manifest.json",
    "reference/official-dspark-config.json",
    "niah-results-eagle.json",
    "niah-results-dspark.json",
    "perf-summary-eagle.json",
    "perf-summary-dspark.json",
    "perf-summary-eagle-thinkon.json",
    "perf-summary-dspark-thinkon.json",
    "perf-summary-dflash2.json",
    "quality-200-sglang-dspark.json",
    "quality-200-sglang-dflash2.json",
    "reference/think-on-extra.json",
    "docker/Dockerfile.dflash2",
)


def test_required_files_exist() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    assert not missing, missing


def test_no_private_host_paths() -> None:
    hits = []
    skip = {".git", "__pycache__", ".pytest_cache"}
    skip_files = {"tests/test_public_tree.py"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in skip for part in path.parts):
            continue
        rel = str(path.relative_to(ROOT))
        if rel in skip_files or path.suffix in {".pyc", ".safetensors", ".jsonl"}:
            continue
        text = path.read_text(errors="ignore")
        for needle in FORBIDDEN:
            if needle in text:
                hits.append(f"{rel}:{needle}")
    assert not hits, hits


def test_quality_200_identity() -> None:
    import hashlib

    raw = (ROOT / "artifacts/quality-200.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "ca35650e0bf4c9997772276c15a7116afd553a305b10c88182f52f050b76e066"
    assert sum(1 for _ in raw.splitlines() if _.strip()) == 200


def test_readme_documents_click_run() -> None:
    text = (ROOT / "README.md").read_text()
    for needle in (
        "scripts/serve.sh dspark",
        "scripts/serve.sh dflash2",
        "scripts/click_run_dspark.sh",
        "scripts/click_run_eagle.sh",
        "scripts/click_run_dflash2.sh",
        "DSparkDraftModel",
        "sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6",
        "quality-200.jsonl",
        "19 × 23",
        "Qwen3DSparkModel",
        "run_perf_suite.sh",
        "Prerequisites",
        "huggingface-cli",
    ):
        assert needle in text, needle


def test_serve_script_covers_official_profiles() -> None:
    text = (ROOT / "scripts/serve.sh").read_text()
    for needle in ("ar)", "eagle)", "dspark)", "dflash2)", "DSPARK", "DFLASH", "qwen3_coder", "flashinfer", "sha256:3c0abdf4"):
        assert needle in text, needle
    assert "Qwen3DSparkModel" not in text


def test_click_run_fails_closed_on_ready_timeout() -> None:
    for name in ("click_run_dspark.sh", "click_run_eagle.sh"):
        text = (ROOT / "scripts" / name).read_text()
        assert "READY_TIMEOUT" in text, name
        assert "container died during startup" in text, name


def test_official_draft_fixture_is_not_vllm_adapter() -> None:
    cfg = json.loads((ROOT / "reference/official-dspark-config.json").read_text())
    assert cfg["architectures"] == ["DSparkDraftModel"]


def test_niah_both_profiles_are_8_of_8() -> None:
    for name in ("niah-results-eagle.json", "niah-results-dspark.json"):
        rows = json.loads((ROOT / name).read_text())
        assert len(rows) == 8, name
        assert all(r.get("pass") for r in rows), name
        assert all(r.get("answer") == "QWEN38-NIAH-9X4K" for r in rows), name


def test_python_scripts_parse() -> None:
    for path in ROOT.joinpath("scripts").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))


def test_shell_scripts_parse() -> None:
    for path in [
        ROOT / "scripts/serve.sh",
        ROOT / "scripts/click_run_dspark.sh",
        ROOT / "scripts/click_run_eagle.sh",
        ROOT / "scripts/run_perf_suite.sh",
        ROOT / "scripts/run_llama_benchy.sh",
    ]:
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{path.name}: {proc.stderr}"
