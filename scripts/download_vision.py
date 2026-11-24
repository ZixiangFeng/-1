"""Fetch the optional local vision model through the installed Ollama runtime."""
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    bundled = ROOT / "runtime/ollama/ollama"
    binary = str(bundled) if bundled.is_file() else shutil.which("ollama")
    if not binary:
        raise SystemExit("Install and start Ollama before downloading the local vision model")
    env = dict(os.environ)
    env.setdefault("OLLAMA_MODELS", str(ROOT / "models/ollama"))
    subprocess.run([binary, "pull", "qwen3-vl:2b"], env=env, check=True)
