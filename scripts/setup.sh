#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runtime/ollama models data
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/python scripts/setup_embeddings.py
if [ ! -x runtime/ollama/ollama ]; then
  curl --http1.1 -fL --retry 5 -C - -o runtime/ollama-darwin.tgz https://github.com/ollama/ollama/releases/download/v0.34.0/ollama-darwin.tgz
  echo 'dd12b00bcce2d6551178e67ada90d5af9f75bdb54a118b96655250fa3e8ef734  runtime/ollama-darwin.tgz' | shasum -a 256 -c -
  tar -xzf runtime/ollama-darwin.tgz -C runtime/ollama
fi
.venv/bin/python scripts/download_model.py
export PATH="$PWD/runtime/ollama:$PATH"
export OLLAMA_MODELS="$PWD/models/ollama"
bash scripts/start.sh
for attempt in {1..30}; do
  if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then break; fi
  sleep 1
done
ollama create expression-companion -f Modelfile
ollama pull qwen3-vl:2b
bash scripts/start.sh
