#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-"$ROOT_DIR/.venv"}"
LOG_DIR="${LOG_DIR:-"$ROOT_DIR/.run_logs"}"

LMCACHE_SERVER_HOST="${LMCACHE_SERVER_HOST:-127.0.0.1}"
LMCACHE_SERVER_PORT="${LMCACHE_SERVER_PORT:-65432}"
LMCACHE_STORAGE_DIR="${LMCACHE_STORAGE_DIR:-/tmp/lmcache_storage}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
DTYPE="${DTYPE:-half}"
VLLM_PORT="${VLLM_PORT:-8000}"
GUIDED_DECODING_BACKEND="${GUIDED_DECODING_BACKEND:-lm-format-enforcer}"
LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-"$ROOT_DIR/lmcache-vllm-extended/configuration.yaml"}"

START_FRONTEND="${START_FRONTEND:-1}"
STREAMLIT_HOST="${STREAMLIT_HOST:-0.0.0.0}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

PIDS=()

cleanup() {
  local pid

  if [ "${#PIDS[@]}" -eq 0 ]; then
    return
  fi

  echo
  echo "Stopping services..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  wait "${PIDS[@]}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

require_file() {
  local path="$1"

  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

wait_for_port() {
  local name="$1"
  local host="$2"
  local port="$3"
  local timeout_seconds="$4"
  local pid="$5"

  python - "$name" "$host" "$port" "$timeout_seconds" "$pid" <<'PY'
import os
import socket
import sys
import time

name, host, port, timeout_seconds, pid = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
deadline = time.monotonic() + timeout_seconds

while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            print(f"{name} is listening on {host}:{port}")
            raise SystemExit(0)
    except OSError:
        try:
            os.kill(pid, 0)
        except OSError:
            print(f"{name} exited before opening {host}:{port}", file=sys.stderr)
            raise SystemExit(1)
        time.sleep(2)

print(f"Timed out waiting for {name} on {host}:{port}", file=sys.stderr)
raise SystemExit(1)
PY
}

start_service() {
  local name="$1"
  local cwd="$2"
  shift 2

  local log_file="$LOG_DIR/$name.log"
  echo "Starting $name, logging to $log_file"
  (cd "$cwd" && "$@") >"$log_file" 2>&1 &
  PIDS+=("$!")
}

require_file "$VENV_DIR/bin/activate"
require_file "$LMCACHE_CONFIG_FILE"

mkdir -p "$LOG_DIR"
mkdir -p "$LMCACHE_STORAGE_DIR"

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

start_service "lmcache-server" "$ROOT_DIR" \
  python3 -m lmcache_server.server "$LMCACHE_SERVER_HOST" "$LMCACHE_SERVER_PORT" "$LMCACHE_STORAGE_DIR"
wait_for_port "LMCache server" "$LMCACHE_SERVER_HOST" "$LMCACHE_SERVER_PORT" 30 "${PIDS[-1]}"

start_service "vllm" "$ROOT_DIR" \
  env LMCACHE_CONFIG_FILE="$LMCACHE_CONFIG_FILE" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  python "$ROOT_DIR/lmcache-vllm-extended/lmcache_vllm/script.py" serve "$MODEL" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$DTYPE" \
    --port "$VLLM_PORT" \
    --guided-decoding-backend "$GUIDED_DECODING_BACKEND"
wait_for_port "vLLM server" "127.0.0.1" "$VLLM_PORT" 900 "${PIDS[-1]}"

if [ "$START_FRONTEND" = "1" ]; then
  start_service "streamlit" "$ROOT_DIR/lmcache-vllm-extended/frontend" \
    streamlit run frontend.py \
      --server.address "$STREAMLIT_HOST" \
      --server.port "$STREAMLIT_PORT"
  wait_for_port "Streamlit frontend" "127.0.0.1" "$STREAMLIT_PORT" 120 "${PIDS[-1]}"
fi

echo
echo "All services are running."
echo "vLLM API: http://127.0.0.1:$VLLM_PORT"
if [ "$START_FRONTEND" = "1" ]; then
  echo "Frontend: http://127.0.0.1:$STREAMLIT_PORT"
fi
echo "Logs are in: $LOG_DIR"
echo "Press Ctrl+C to stop everything."

wait -n "${PIDS[@]}"
