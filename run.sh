#!/usr/bin/env bash
set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$BASE_DIR/venv/bin/python" "$BASE_DIR/app.py" "$@"
