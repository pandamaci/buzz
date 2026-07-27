#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  printf 'Usage: %s [--cpu-only|--cuda-only|--configure-hf-token]\n' "$0"
  printf '  --configure-hf-token  Validate and save a Hugging Face read token, then exit.\n'
}

prepare_checkout() {
  local git_state

  if command -v git >/dev/null 2>&1; then
    if git_state=$(git rev-parse --is-inside-work-tree 2>/dev/null) && [[ "$git_state" == true ]]; then
      if ! git submodule update --init --recursive; then
        printf 'Unable to initialize Buzz git submodules. Check Git access and network connectivity, then run:\n' >&2
        printf '  git submodule update --init --recursive\n' >&2
        exit 1
      fi
    elif [[ -e .git ]]; then
      printf 'Unable to inspect the Buzz git worktree. Check that Git is installed and this checkout is valid.\n' >&2
      exit 1
    fi
  elif [[ -e .git ]]; then
    printf 'Buzz is a Git worktree, but Git is not installed. Install Git and rerun this launcher.\n' >&2
    exit 1
  fi
}

preflight_native_tools() {
  local tool
  local missing=()
  local required_tools=(cc c++ make cmake)

  for tool in "${required_tools[@]}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      missing+=("$tool")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    printf 'Missing native build tool(s): %s\n' "${missing[*]}" >&2
    printf 'Install a C/C++ toolchain, make, and CMake with your OS package manager, then rerun ./start.sh.\n' >&2
    printf 'Examples: build-essential + cmake (Debian/Ubuntu), Development Tools + cmake (Fedora/RHEL), or Xcode Command Line Tools + cmake (macOS).\n' >&2
    exit 1
  fi
}

preflight_runtime() {
  local ldconfig_output
  local qt_plugin
  local ldd_output
  local line
  local library
  local missing_libraries=()

  if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is required but was not found on PATH.\n' >&2
    printf 'Install it with pipx, ensure its bin directory is on PATH, then restart your shell:\n' >&2
    printf '  pipx install uv\n' >&2
    printf '  pipx ensurepath\n' >&2
    printf 'Verify the installation with: command -v uv\n' >&2
    exit 1
  fi

  if ! command -v ffmpeg >/dev/null 2>&1; then
    printf 'ffmpeg is required to transcribe imported audio and video files but was not found on PATH.\n' >&2
    printf 'Install it with:\n' >&2
    printf '  sudo apt-get install ffmpeg\n' >&2
    exit 1
  fi

  if [[ "$(uname -s)" == Linux ]]; then
    if command -v ldconfig >/dev/null 2>&1; then
      ldconfig_output="$(ldconfig -p 2>/dev/null || true)"
      if [[ "$ldconfig_output" != *libportaudio.so.2* ]]; then
        printf 'PortAudio runtime library libportaudio.so.2 was not found.\n' >&2
        printf 'Install it with:\n' >&2
        printf '  sudo apt-get install libportaudio2 libpulse0 libasound2\n' >&2
        exit 1
      fi
    fi

    if ! command -v ldd >/dev/null 2>&1; then
      printf 'ldd is required to check the Qt XCB platform plugin dependencies.\n' >&2
      printf 'Install the package that provides ldd, then rerun ./start.sh.\n' >&2
      exit 1
    fi

    qt_plugin=".venv-${1}/lib/python3.12/site-packages/PyQt6/Qt6/plugins/platforms/libqxcb.so"
    if [[ ! -f "$qt_plugin" ]]; then
      printf 'The selected %s environment does not contain PyQt6 Qt platform plugin libqxcb.so.\n' "$1" >&2
      printf 'Install the selected locked environment before launching Buzz, then rerun ./start.sh.\n' >&2
      printf '  UV_PROJECT_ENVIRONMENT=.venv-%s uv sync --locked --exact --no-default-groups --extra %s\n' "$1" "$1" >&2
      exit 1
    fi

    ldd_output="$(ldd "$qt_plugin" 2>&1 || true)"
    while IFS= read -r line; do
      if [[ "$line" == *'not found'* ]]; then
        library="${line%% =>*}"
        library="${library#"${library%%[![:space:]]*}"}"
        library="${library%"${library##*[![:space:]]}"}"
        missing_libraries+=("$library")
      fi
    done <<< "$ldd_output"

    if (( ${#missing_libraries[@]} > 0 )); then
      printf 'Qt XCB platform plugin libqxcb.so has unresolved dependencies:\n' >&2
      printf '  %s\n' "${missing_libraries[@]}" >&2
      printf 'Install the Ubuntu Qt/XCB runtime packages with:\n' >&2
      printf '  sudo apt-get install --no-install-recommends libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-shape0 libxcb-cursor0\n' >&2
      exit 1
    fi
  fi
}

offer_deepfilternet_prerequisites() {
  local os_id
  local apt_runner
  local answer

  if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    return
  fi

  if [[ ! -t 0 || ! -r /dev/tty || ! -r /etc/os-release ]]; then
    return
  fi

  os_id=''
  . /etc/os-release
  os_id="${ID:-}"
  case "$os_id" in
    ubuntu|debian) ;;
    *) return ;;
  esac

  if ! command -v apt-get >/dev/null 2>&1; then
    printf 'DeepFilterNet is optional; apt-get is unavailable, so its Rust build prerequisites were not installed.\n' >&2
    return
  fi

  if [[ "$EUID" -eq 0 ]]; then
    apt_runner=(apt-get)
  elif command -v sudo >/dev/null 2>&1; then
    apt_runner=(sudo apt-get)
  else
    printf 'DeepFilterNet is optional; sudo is unavailable, so its Rust build prerequisites were not installed.\n' >&2
    return
  fi

  printf 'DeepFilterNet noise reduction is optional. Cargo and/or Rust are missing, so its build prerequisites can be installed with apt.\n' >&2
  printf 'Install cargo and rustc now? [y/N] ' >&2
  if ! IFS= read -r answer </dev/tty; then
    printf '\n' >&2
    return
  fi

  case "$answer" in
    [yY]|[yY][eE][sS])
      if ! "${apt_runner[@]}" install --no-install-recommends cargo rustc; then
        printf 'Unable to install the optional DeepFilterNet prerequisites; continuing without them.\n' >&2
      fi
      ;;
    *)
      printf 'Skipping the optional DeepFilterNet prerequisites; Buzz will continue without DeepFilterNet.\n' >&2
      ;;
  esac
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

prepare_checkout
preflight_native_tools
preflight_runtime "$mode"

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

offer_deepfilternet_prerequisites

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
