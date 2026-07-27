#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  printf 'Usage: %s [--cpu-only|--cuda-only|--configure-hf-token]\n' "$0"
  printf '  --configure-hf-token  Validate and save a Hugging Face read token, then exit.\n'
}

if (( $# > 1 )); then
  printf 'Only one launcher option may be supplied.\n' >&2
  usage >&2
  exit 2
fi

mode=cpu
case "${1:-}" in
  "") ;;
  --cpu-only) mode=cpu ;;
  --cuda-only) mode=cuda ;;
  --configure-hf-token) ;;
  --help|-h)
    usage
    exit 0
    ;;
  *)
    printf 'Unknown option: %s\n' "$1" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "${1:-}" == --configure-hf-token ]]; then
  export HF_HOME="${HOME}/.cache/Buzz"
  if [[ ! -t 0 || ! -r /dev/tty ]]; then
    printf 'Hugging Face token configuration requires an interactive terminal.\n' >&2
    exit 2
  fi

  printf 'Hugging Face read token (input hidden): ' >&2
  if ! IFS= read -r -s token </dev/tty; then
    printf '\nUnable to read the Hugging Face token.\n' >&2
    exit 2
  fi
  printf '\n' >&2
  if [[ -z "$token" ]]; then
    printf 'A non-empty Hugging Face token is required.\n' >&2
    exit 2
  fi

  if ! printf '%s\n' "$token" | UV_PROJECT_ENVIRONMENT=.venv-cpu uv run --locked --exact --no-default-groups --extra cpu python -c '
import sys
from huggingface_hub import login

token = sys.stdin.readline().rstrip("\n")
login(token=token, add_to_git_credential=False)
'; then
    unset token
    printf 'Hugging Face login failed; no token was saved by Buzz.\n' >&2
    exit 1
  fi
  unset token
  printf 'Hugging Face login validated and saved in the local Hugging Face cache.\n'
  exit 0
fi

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
  uv run --locked --exact --no-default-groups --extra cuda python scripts/ensure_hungarian_whispercpp_models.py
  exec uv run --locked --exact --no-default-groups --extra cuda buzz
else
  export UV_PROJECT_ENVIRONMENT=.venv-cpu
  uv run --locked --exact --no-default-groups --extra cpu python scripts/ensure_hungarian_whispercpp_models.py
  exec uv run --locked --exact --no-default-groups --extra cpu buzz
fi
