#!/usr/bin/env bash
set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$BASE_DIR/venv"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "[ERROR] 找不到虚拟环境: $PY" >&2
  echo "        请先创建 venv 并安装依赖: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

MODE="https"
for arg in "$@"; do
  case "$arg" in
    --http|--no-https) MODE="http" ;;
    --https)            MODE="https" ;;
    -h|--help)
      echo "Usage: $0 [--http|--https]"
      echo "  --http   以 HTTP  模式启动 (默认 http://0.0.0.0:1111)"
      echo "  --https  以 HTTPS 模式启动 (默认,https://0.0.0.0:1111,自签名证书)"
      exit 0
      ;;
  esac
done

export PYTHONUNBUFFERED=1

if [[ "$MODE" == "https" ]]; then
  echo "[start]  HTTPS 模式 → https://0.0.0.0:1111 (自签名证书,首次访问浏览器需信任)"
  exec "$PY" "$BASE_DIR/app.py" --https
else
  echo "[start]  HTTP  模式 → http://0.0.0.0:1111"
  exec "$PY" "$BASE_DIR/app.py"
fi
