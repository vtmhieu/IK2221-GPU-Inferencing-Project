#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-"$ROOT_DIR/.venv"}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

echo "Project root: $ROOT_DIR"
echo "Virtual environment: $VENV_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was installed, but it is not on PATH. Restart your shell or add uv to PATH." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

uv pip install -r "$ROOT_DIR/lmcache-vllm-extended/requirements.txt"
uv pip install matplotlib
uv pip install -e "$ROOT_DIR/lmcache-vllm-extended"
uv pip install -e "$ROOT_DIR/LMCache"
uv pip install -e "$ROOT_DIR/lmcache-server"

echo
echo "Setup complete. Activate the environment with:"
echo "  source \"$VENV_DIR/bin/activate\""
