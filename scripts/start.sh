#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p data
export PATH="$PWD/runtime/ollama:/opt/homebrew/bin:$PATH"
export OLLAMA_MODELS="$PWD/models/ollama"
cloud_enabled=$(.venv/bin/python -c 'import cloud_api; print(int(bool(cloud_api.load_config("data").get("enabled"))))')
if [ "$cloud_enabled" = "0" ] && ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
  .venv/bin/python - <<'START_OLLAMA'
import os, pathlib, subprocess
root = pathlib.Path.cwd()
env = dict(os.environ, OLLAMA_NO_CLOUD="true", OLLAMA_HOST="127.0.0.1:11434", OLLAMA_MAX_LOADED_MODELS="1")
with open("data/ollama.log", "ab") as log:
    p = subprocess.Popen([str(root / "runtime/ollama/ollama"), "serve"], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
pathlib.Path("data/ollama.pid").write_text(str(p.pid))
START_OLLAMA
fi
if ! curl -sf http://127.0.0.1:8765/api/health >/dev/null; then
  .venv/bin/python - <<'START_SERVER'
import pathlib, subprocess, sys
with open("data/server.log", "ab") as log:
    p = subprocess.Popen([sys.executable, "server.py"], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
pathlib.Path("data/server.pid").write_text(str(p.pid))
START_SERVER
fi
sleep 2
if command -v open >/dev/null; then
  open "http://127.0.0.1:8765/research#key=$(cat data/admin.key)"
fi
