#!/usr/bin/env bash
# mc — terminal client for the miniclosedai-llm dashboard.
# Thin wrapper: runs cli.py with the project venv if present, else system python3.
# The CLI is stdlib-only, so either works. Symlink this into your PATH if you like:
#   ln -s "$(pwd)/mc" ~/.local/bin/mc
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$DIR/.venv/bin/python" ]; then
  PY="$DIR/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi
exec "$PY" "$DIR/cli.py" "$@"
