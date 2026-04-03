#!/usr/bin/env python3
"""
One-shot setup: .env, PDFs from the official sample repo, optional Ollama pull, build RAG indices.
Run from anywhere: python bootstrap.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
ENV_EX = os.path.join(ROOT, ".env.example")
ENV = os.path.join(ROOT, ".env")

ASSETS = {
    "FY24_Q4_Consolidated_Financial_Statements.pdf": (
        "https://raw.githubusercontent.com/jason79461385/Assignment-3/main/data/"
        "FY24_Q4_Consolidated_Financial_Statements.pdf"
    ),
    "tsla-20241231-gen.pdf": (
        "https://raw.githubusercontent.com/jason79461385/Assignment-3/main/data/"
        "tsla-20241231-gen.pdf"
    ),
}


def log(msg: str) -> None:
    print(f"[bootstrap] {msg}", flush=True)


def ensure_env() -> None:
    if os.path.isfile(ENV):
        log(".env already exists; not overwriting.")
        return
    if not os.path.isfile(ENV_EX):
        log("warning: .env.example missing; skipping .env")
        return
    shutil.copy(ENV_EX, ENV)
    log("Created .env from .env.example")


def download_if_needed(name: str, url: str) -> None:
    os.makedirs(DATA, exist_ok=True)
    dest = os.path.join(DATA, name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
        log(f"Already have {name} ({os.path.getsize(dest)} bytes); skip download.")
        return
    log(f"Downloading {name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "assignment3-bootstrap/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)
    log(f"Saved {name} ({len(data)} bytes)")


def maybe_pull_ollama() -> None:
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        log("ollama CLI not found; skip model pull (install from https://ollama.com).")
        return
    # Load model name from .env if present
    model = "llama3.2"
    if os.path.isfile(ENV):
        with open(ENV, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OLLAMA_MODEL=") and not line.startswith("#"):
                    model = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    log(f"Running: ollama pull {model} (needs ollama serve running)")
    try:
        r = subprocess.run(
            [ollama_bin, "pull", model],
            cwd=ROOT,
            timeout=3600,
        )
        if r.returncode != 0:
            log(f"ollama pull exited {r.returncode}; continue anyway.")
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        log("ollama pull timed out; continue anyway.")
    except Exception as e:
        log(f"ollama pull failed: {e}; continue anyway.")


def build_indices() -> None:
    log("Running build_rag.py ...")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "build_rag.py")], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)
    log("build_rag.py finished.")


def main() -> None:
    os.chdir(ROOT)
    ensure_env()
    for fname, url in ASSETS.items():
        try:
            download_if_needed(fname, url)
        except urllib.error.HTTPError as e:
            log(f"HTTP error downloading {fname}: {e}")
            sys.exit(1)
        except Exception as e:
            log(f"Failed to download {fname}: {e}")
            sys.exit(1)
    maybe_pull_ollama()
    build_indices()
    log("Done. Run: EVAL_MODE=GRAPH python evaluator.py")


if __name__ == "__main__":
    main()
