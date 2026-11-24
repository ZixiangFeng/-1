"""Complete pending vision setup and real smoke test, without requiring an open terminal."""

import json, pathlib, subprocess, time, urllib.request

root = pathlib.Path(__file__).resolve().parents[1]
manifest = root / "models/ollama/manifests/registry.ollama.ai/library/qwen3-vl/2b"
while not manifest.exists():
    time.sleep(10)
# Register through Ollama's normal pull path; already verified local blobs are reused.
r = subprocess.run(
    [str(root / "runtime/ollama/ollama"), "pull", "qwen3-vl:2b"],
    cwd=root,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
if r.returncode:
    raise RuntimeError(r.stdout[-1000:])
print("Vision model registered", flush=True)
r = subprocess.run(
    [str(root / ".venv/bin/python"), "scripts/verify_images_live.py"], cwd=root
)
print("Live image verification exit:", r.returncode, flush=True)
raise SystemExit(r.returncode)
