#!/bin/bash
set -e
cd "$(dirname "$0")/.."
if [ -f data/server.pid ]; then
  pid=$(cat data/server.pid)
  if ps -p "$pid" -o command= | grep -iq 'python server.py'; then kill "$pid"; fi
  rm -f data/server.pid
fi
