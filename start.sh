#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if (( $# > 1 )); then
  printf 'Only one launcher option may be supplied.\nUsage: %s [--cpu-only|--cuda-only]\n' "$0" >&2
  exit 2
fi

mode=cpu
case "${1:-}" in
  "") ;;
  --cpu-only) mode=cpu ;;
  --cuda-only) mode=cuda ;;
  --help|-h)
    printf 'Usage: %s [--cpu-only|--cuda-only]\n' "$0"
    exit 0
    ;;
  *)
    printf 'Unknown option: %s\nUsage: %s [--cpu-only|--cuda-only]\n' "$1" "$0" >&2
    exit 2
    ;;
esac

if [[ "$mode" == cuda ]]; then
  unset BUZZ_FORCE_CPU
  if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    printf 'CUDA mode requires Linux x86_64.\n' >&2
    exit 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    printf 'CUDA mode requires a working NVIDIA driver/GPU (nvidia-smi failed); no CUDA environment was created.\n' >&2
    exit 1
  fi
  if ! UV_PROJECT_ENVIRONMENT=.venv-cuda uv run --locked --exact --no-default-groups --extra cuda python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    printf 'CUDA mode requires a working NVIDIA driver and CUDA-enabled PyTorch (torch.cuda.is_available() is false).\n' >&2
    exit 1
  fi
else
  export BUZZ_FORCE_CPU=true
fi
export BUZZ_EXECUTION_MODE="$mode"
export BUZZ_WHISPERCPP_N_THREADS=8
export BUZZ_DISABLE_UPDATE_CHECK=true
export BUZZ_MODEL_ROOT="${HOME}/.cache/Buzz/models"
export HF_HOME="${HOME}/.cache/Buzz"

if [[ "$mode" == cuda ]]; then
  export UV_PROJECT_ENVIRONMENT=.venv-cuda
  exec uv run --locked --exact --no-default-groups --extra cuda buzz
else
  export UV_PROJECT_ENVIRONMENT=.venv-cpu
  exec uv run --locked --exact --no-default-groups --extra cpu buzz
fi
