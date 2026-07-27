# Buzz setup and launch

These instructions target Ubuntu/Linux. Buzz requires Python 3.12 (`>=3.12,<3.13`) and [uv](https://docs.astral.sh/uv/).

## Prerequisites

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends \
  build-essential cmake pkg-config curl python3.12 python3.12-dev python3.12-venv \
  libyaml-dev libtbb-dev libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
  libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 \
  libxcb-shape0 libxcb-cursor0 libportaudio2 gettext libpulse0 ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Initialize submodules, then install the locked CPU environment:

```bash
git submodule update --init --recursive
UV_PROJECT_ENVIRONMENT=.venv-cpu uv sync --locked --exact --no-default-groups --extra cpu
```

## Launch

CPU is the default and uses `.venv-cpu`:

```bash
./start.sh
./start.sh --cpu-only
```

CUDA uses a separate `.venv-cuda` and performs all checks before opening the GUI:

```bash
./start.sh --cuda-only
```

CUDA requires Linux x86_64, an NVIDIA GPU with a working driver, and a compatible CUDA 12.9 PyTorch installation. The CUDA extra installs PyTorch CUDA wheels and their matching NVIDIA runtime dependencies; it does not enable CUDA on macOS or other architectures. CUDA mode accelerates CUDA-capable Python backends such as Faster Whisper, Transformers, and OpenAI Whisper. This checkout's bundled Whisper.cpp is built CPU-only, so its file/live paths remain CPU-only even in CUDA mode. A CUDA launch fails clearly when `torch.cuda.is_available()` is false. CPU and CUDA environments are intentionally separate; do not mix their packages.

The launcher sets `BUZZ_EXECUTION_MODE`, which overrides the saved **Disable GPU** preference: CPU mode forces CPU and hides CUDA, while CUDA mode is not defeated by a previously saved force-CPU setting. Only CUDA-capable Python backends follow the selected mode; Whisper.cpp remains CPU-only. macOS remains supported by the regular project dependencies, but this launcher’s CUDA mode is Linux-only.

Models are downloaded from **Help → Preferences → Models**. For CPU, start with Whisper.cpp `tiny` or `base`; larger models need more memory and are slower. Local transcription needs no API key.

### First-run Hungarian model provisioning

The launcher provisions the pinned Hungarian Whisper.cpp models before opening Buzz. If Hugging Face asks for authentication, create a Hugging Face [read access token](https://huggingface.co/settings/tokens) and configure it before the first launch:

```bash
./start.sh --configure-hf-token
./start.sh
```

The first command prompts without echoing the token, validates it with the locked CPU environment's official `huggingface_hub` login API, saves the login in `~/.cache/Buzz`, and exits. It does not write the token to this checkout or to Git. The provisioner automatically uses that saved login; no model needs to be downloaded first. To avoid saving a login, provide the token only in the process environment instead (do not put it on a command line):

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN; printf '\n'
export HF_TOKEN
./start.sh
unset HF_TOKEN
```

Remove the saved login with:

```bash
HF_HOME="$HOME/.cache/Buzz" UV_PROJECT_ENVIRONMENT=.venv-cpu \
  uv run --locked --exact --no-default-groups --extra cpu python -c \
  'from huggingface_hub import logout; logout()'
```

Revoke the token itself from the Hugging Face [token settings](https://huggingface.co/settings/tokens). The bundled Hungarian Base fine-tune has an upstream **non-commercial restriction**; do not treat it as cleared for commercial use.

## Troubleshooting

- `uv: command not found`: restart the shell after installing uv.
- Qt/display or audio errors: run from a graphical desktop and install the prerequisite packages above.
- CUDA mode errors: verify `nvidia-smi`, the NVIDIA driver, and that the launcher is running on Linux x86_64.
- Whisper.cpp build errors: confirm submodules, CMake, and a C/C++ toolchain are installed.
