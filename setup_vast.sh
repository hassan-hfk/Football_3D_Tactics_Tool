#!/usr/bin/env bash
# =============================================================================
# FORMA 3D — Linux / CUDA Setup Script (vast.ai, Lambda, RunPod, any Linux+CUDA box)
#
# Place this file inside the app/ folder, then run:
#
#   cd app
#   chmod +x setup_vast.sh
#   ./setup_vast.sh
#
# Tested on:
#   - vast.ai instances using image  pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
#   - Lambda Cloud H100/A100/A6000 instances
#   - Bare Ubuntu 22.04 + CUDA 11.8/12.1/12.4
#   - Ubuntu 22.04 CPU-only (no NVIDIA driver) — falls back to CPU PyTorch
#
# Requirements:
#   - Ubuntu 20.04 / 22.04 (or any Debian-derived distro)
#   - Python 3.10 or 3.11
#   - Internet access for pip + git + apt
#   - Either: NVIDIA GPU with driver R525+ (CUDA 12.x) or R470+ (CUDA 11.x)
#     Or:    nothing — script falls back to CPU PyTorch
#
# What this does that setup_mac.sh does NOT:
#   - Installs CUDA PyTorch wheels (matched to detected CUDA version)
#   - Installs onnxruntime-gpu instead of onnxruntime-silicon
#   - Builds pytorch3d from source against the live torch in the venv,
#     because prebuilt wheels (PyTorch's CDN + community indexes) have
#     repeatedly produced installs whose _C extension fails to load at
#     runtime on real host+driver combos (confirmed by user). Source
#     build is slower (5-10 min CUDA, 15-25 min CPU) but the resulting
#     _C.so cannot be ABI-mismatched against this venv's torch.
#   - Uses apt instead of brew for system deps
#
# CUDA-specific note: pytorch3d's source build needs nvcc, which is NOT in
# the pytorch/pytorch :runtime images. On vast.ai pick a DEVEL image:
#   pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
# The runtime image will work for everything except this step, which will
# fail with a clear error in Step 4 telling you what to do.
#
# Idempotent: re-running skips anything already installed.
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}${BOLD}──${RESET} $*"; }
success() { echo -e "  ${GREEN}✓${RESET}  $*"; }
warn()    { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
err()     { echo -e "  ${RED}✗  ERROR:${RESET} $*"; }
die()     { err "$*"; exit 1; }

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  FORMA 3D — Linux / CUDA Setup (vast.ai etc.)        ${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo ""

# ── Paths (everything relative to app/ where this script lives) ───────────────
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/.venv"
CKPT_DIR="$APP_DIR/GVHMR/inputs/checkpoints"

# ── Sudo wrapper (vast.ai containers run as root with no sudo binary) ─────────
SUDO=""
if [[ $EUID -ne 0 ]]; then
    if command -v sudo &>/dev/null; then
        SUDO="sudo"
    else
        warn "Not running as root and no sudo found — apt installs may fail"
    fi
fi

# =============================================================================
# STEP 0 — Preflight checks
# =============================================================================
info "Step 0 — Preflight checks"

# OS
if [[ ! -f /etc/os-release ]]; then
    die "Cannot determine OS — /etc/os-release missing. This script targets Debian/Ubuntu."
fi
source /etc/os-release
case "${ID:-}" in
    ubuntu|debian)
        success "OS: $PRETTY_NAME"
        ;;
    *)
        warn "OS: $PRETTY_NAME — script is tested on Ubuntu/Debian, may need adjustment"
        ;;
esac

# Architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" || "$ARCH" == "aarch64" ]]; then
    success "Architecture: $ARCH"
else
    warn "Architecture: $ARCH — untested"
fi

# git
if ! command -v git &>/dev/null; then
    warn "git not found — will install via apt"
fi

# =============================================================================
# STEP 0a — System packages (apt)
# =============================================================================
info "Step 0a — System packages via apt"

# Skip apt entirely if we're not root AND we have no sudo — common in
# locked-down environments. The user is expected to have installed these.
if [[ -z "$SUDO" && $EUID -ne 0 ]]; then
    warn "Skipping apt — no privileges. Required packages:"
    warn "  python3 python3-venv python3-tk python3-dev"
    warn "  ffmpeg git build-essential libgl1 libglib2.0-0"
else
    echo "  Updating apt index ..."
    $SUDO apt-get update -qq

    # Core build / Python tooling
    # - python3-tk: GVHMR imports tkinter (same reason as Mac's python-tk@X)
    # - libgl1, libglib2.0-0: OpenCV import on headless Ubuntu
    # - build-essential, python3-dev: pytorch3d source-build fallback
    # - ffmpeg: /transcode-to-mp4 endpoint in main.py (imageio-ffmpeg has
    #   its own binary but main._find_ffmpeg() looks at PATH first)
    echo "  Installing: python3-venv python3-tk python3-dev ffmpeg git build-essential libgl1 libglib2.0-0 ..."
    $SUDO apt-get install -y -qq \
        python3 python3-venv python3-tk python3-dev \
        ffmpeg git build-essential \
        libgl1 libglib2.0-0 \
        wget curl ca-certificates
    success "apt packages installed"
fi

# Python — prefer 3.11, fall back to 3.10, then generic python3
PYTHON=""
for candidate in python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$(command -v "$candidate")"
        break
    fi
done
[[ -z "$PYTHON" ]] && die "Python 3.10+ not found.\nInstall:  $SUDO apt install python3.11 python3.11-venv"

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

[[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]] && \
    die "Python $PY_VER found but 3.10+ is required."
[[ "$PY_MINOR" -ge 12 ]] && \
    warn "Python $PY_VER — 3.12+ is untested with GVHMR. 3.11 is safest."

success "Python: $PYTHON ($PY_VER)"
success "git: $(git --version)"

# =============================================================================
# STEP 0b — GPU / CUDA detection
# =============================================================================
info "Step 0b — GPU / CUDA detection"

GPU_MODE="cpu"        # "cuda" or "cpu"
CUDA_TAG=""           # "cu118" | "cu121" | "cu124"  (PyTorch wheel suffix)
CUDA_VER=""           # "11.8" | "12.1" | "12.4"  (display only)

if command -v nvidia-smi &>/dev/null; then
    # Parse "CUDA Version: 12.1" from the nvidia-smi header
    # On older drivers the line is "CUDA Version: 11.8". Some images print
    # nothing — fall back to nvcc if present.
    NVSMI_OUT=$(nvidia-smi 2>/dev/null || echo "")
    DRIVER_VER=$(echo "$NVSMI_OUT" | awk -F'Driver Version: ' 'NR==3 {split($2, a, " "); print a[1]}' || true)
    DETECTED_CUDA=$(echo "$NVSMI_OUT" | awk -F'CUDA Version: ' 'NR==3 {split($2, a, " "); print a[1]}' || true)

    if [[ -z "$DETECTED_CUDA" ]] && command -v nvcc &>/dev/null; then
        DETECTED_CUDA=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || true)
    fi

    if [[ -n "$DETECTED_CUDA" ]]; then
        GPU_MODE="cuda"
        CUDA_VER="$DETECTED_CUDA"
        # Pick the closest PyTorch wheel index. PyTorch ships wheels for
        # CUDA 11.8, 12.1, and 12.4. Newer driver/runtime is forward-compatible
        # with older toolkit-built wheels, so we round DOWN to the nearest
        # PyTorch-supported version.
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        if   [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 4 ]]; then CUDA_TAG="cu124"
        elif [[ "$CUDA_MAJOR" -ge 12 ]];                       then CUDA_TAG="cu121"
        elif [[ "$CUDA_MAJOR" -ge 11 && "$CUDA_MINOR" -ge 8 ]]; then CUDA_TAG="cu118"
        else
            warn "Detected CUDA $CUDA_VER — older than 11.8. Forcing cu118 wheels; upgrade your driver if PyTorch fails to init."
            CUDA_TAG="cu118"
        fi
        success "GPU detected — driver $DRIVER_VER, CUDA $CUDA_VER, using wheels: $CUDA_TAG"

        GPU_NAME=$(echo "$NVSMI_OUT" | awk '/[0-9]+%/ {gsub(/^ +/, "", $0); print; exit}' || echo "")
        [[ -n "$GPU_NAME" ]] && success "GPU: $GPU_NAME"
    else
        warn "nvidia-smi exists but did not report a CUDA version — falling back to CPU"
    fi
else
    warn "No nvidia-smi — installing CPU-only PyTorch (pipeline will work but is much slower)"
fi

# nvcc presence is required for pytorch3d's CUDA source build (Step 4).
# Track it as a separate flag so Step 4 can fail fast with a useful message
# instead of letting setup.py run for several minutes and then die with a
# cryptic cpp_extension error.
HAS_NVCC=0
if [[ "$GPU_MODE" == "cuda" ]]; then
    if command -v nvcc &>/dev/null; then
        HAS_NVCC=1
        NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "?")
        success "nvcc found (CUDA toolkit $NVCC_VER) — pytorch3d source build can use CUDA"
        # Derive CUDA_HOME from nvcc location if not already set.
        # setup.py inspects this env var to find toolkit headers/libs.
        if [[ -z "${CUDA_HOME:-}" ]]; then
            CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
            export CUDA_HOME
            echo "  CUDA_HOME = $CUDA_HOME"
        fi
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        HAS_NVCC=1
        export PATH="/usr/local/cuda/bin:$PATH"
        export CUDA_HOME=/usr/local/cuda
        NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "?")
        success "nvcc found at /usr/local/cuda/bin (CUDA toolkit $NVCC_VER) — added to PATH"
    else
        warn "nvcc not found — pytorch3d source build will fail at Step 4."
        warn "  Most likely cause: you're on a *runtime* CUDA image (libcudart only),"
        warn "  not a *devel* one. On vast.ai recreate the instance with:"
        warn "    pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel"
        warn "  Or on a bare host: $SUDO apt install -y nvidia-cuda-toolkit"
        warn "  (may install a system CUDA differing from your driver's CUDA — prefer the devel image)"
    fi
fi

# =============================================================================
# STEP 1 — Virtual environment
# =============================================================================
info "Step 1 — Virtual environment"

# A venv carried over from another machine (e.g. the Mac dev box, in a zipped
# app folder shipped to vast) is unusable here — its python binary is Mach-O
# instead of ELF, and pyvenv.cfg's home= path points to /opt/homebrew or
# similar which doesn't exist on Linux. Detect that and rebuild, otherwise
# every subsequent step fails with confusing "file not found" errors.
if [[ -d "$VENV_DIR" ]]; then
    if [[ -x "$VENV_DIR/bin/python" ]] && "$VENV_DIR/bin/python" --version &>/dev/null; then
        success "Virtual environment already exists — reusing"
    else
        warn ".venv exists but its python is not runnable on this machine"
        warn "  (likely shipped from a different OS / arch). Rebuilding ..."
        rm -rf "$VENV_DIR"
        "$PYTHON" -m venv "$VENV_DIR"
        success "Virtual environment rebuilt"
    fi
else
    echo "  Creating .venv inside app/ ..."
    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PIP="$VENV_DIR/bin/pip"
PYTHON3="$VENV_DIR/bin/python"

"$PIP" install --upgrade pip setuptools wheel --quiet
success "pip / setuptools / wheel upgraded"

# Verify tkinter inside the venv (venv inherits system Python's _tkinter.so)
if "$PYTHON3" -c "import _tkinter" 2>/dev/null; then
    success "tkinter (_tkinter) importable inside venv"
else
    warn "tkinter not importable inside venv. If GVHMR fails with '_tkinter' errors, run:"
    warn "  $SUDO apt install python3-tk"
    warn "  then recreate the venv: rm -rf .venv && ./setup_vast.sh"
fi

# =============================================================================
# STEP 2 — PyTorch (CUDA or CPU)
# =============================================================================
info "Step 2 — PyTorch ($GPU_MODE)"

NEED_TORCH_INSTALL=1
if "$PYTHON3" -c "
import torch, sys
major, minor = (int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
ok = (major, minor) >= (2, 1)
if ok and '$GPU_MODE' == 'cuda':
    ok = torch.cuda.is_available()
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    TORCH_VER=$("$PYTHON3" -c "import torch; print(torch.__version__)")
    CUDA_OK=$("$PYTHON3" -c "import torch; print(torch.cuda.is_available())")
    success "PyTorch $TORCH_VER already installed (cuda.is_available=$CUDA_OK) — skipping"
    NEED_TORCH_INSTALL=0
fi

if [[ $NEED_TORCH_INSTALL -eq 1 ]]; then
    if [[ "$GPU_MODE" == "cuda" ]]; then
        echo "  Installing PyTorch + torchvision with $CUDA_TAG wheels ..."
        "$PIP" install --quiet \
            torch torchvision \
            --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
    else
        echo "  Installing PyTorch + torchvision (CPU wheels) ..."
        "$PIP" install --quiet \
            torch torchvision \
            --index-url "https://download.pytorch.org/whl/cpu"
    fi
    success "PyTorch installed"
fi

"$PYTHON3" - <<'PYEOF'
import torch
print(f"  \033[32m✓\033[0m  torch {torch.__version__}")
if torch.cuda.is_available():
    print(f"  \033[32m✓\033[0m  CUDA available — {torch.cuda.get_device_name(0)} "
          f"(capability {'.'.join(map(str, torch.cuda.get_device_capability(0)))})")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print(f"  \033[32m✓\033[0m  MPS available")
else:
    print(f"  \033[33m⚠\033[0m  No GPU — CPU mode")
PYEOF

# =============================================================================
# STEP 3 — All pip dependencies
# =============================================================================
info "Step 3 — pip dependencies"

# 1. numpy first — pinned to 1.26.4 exactly.
# onnxruntime is compiled against numpy 1.x ABI; numpy 2.x causes a segfault
# inside onnxruntime_pybind11_state.so at CreateTensor on every platform.
echo "  [1/10] numpy, scipy, shapely, pyyaml, easydict, matplotlib ..."
"$PIP" install "numpy==1.26.4" scipy shapely pyyaml easydict matplotlib --quiet

# 2. FastAPI server
echo "  [2/10] fastapi, uvicorn, python-multipart ..."
"$PIP" install fastapi "uvicorn[standard]" python-multipart --quiet

# 3. OpenCV (headless variant — avoids GUI/Qt deps that break on server boxes)
echo "  [3/10] opencv-python-headless ..."
"$PIP" install opencv-python-headless --quiet

# 4. Video I/O
echo "  [4/10] imageio, imageio-ffmpeg, av ..."
"$PIP" install "imageio[ffmpeg]" imageio-ffmpeg --quiet
"$PIP" install av --quiet \
    || warn "av (PyAV) failed — imageio fallback reader will be used (OK)"

# 5. ffmpeg-python
echo "  [5/10] ffmpeg-python ..."
"$PIP" install ffmpeg-python --quiet

# 6. YOLO + BoT-SORT tracking
echo "  [6/10] ultralytics ..."
"$PIP" install ultralytics --quiet

# 7. Team clustering (ResNet-26 embeddings)
echo "  [7/10] transformers ..."
"$PIP" install transformers --quiet

# 8. ONNX Runtime — GPU build if CUDA available, plain CPU build otherwise.
# Must be installed AFTER numpy==1.26.4 is locked in (ABI compatibility).
echo "  [8/10] onnxruntime ..."
if [[ "$GPU_MODE" == "cuda" ]]; then
    "$PIP" install onnxruntime-gpu --quiet \
        || { warn "onnxruntime-gpu failed — falling back to onnxruntime (CPU)"
             "$PIP" install onnxruntime --quiet; }
else
    "$PIP" install onnxruntime --quiet \
        || warn "onnxruntime install failed"
fi
# Force-reinstall numpy after onnxruntime to ensure nothing pulled it back to 2.x
"$PIP" install "numpy==1.26.4" --force-reinstall --quiet

# 9. GVHMR / SMPL-X stack — order matters
echo "  [9/10] GVHMR stack: colorlog, einops, hydra, lightning, timm, smplx, wis3d ..."
"$PIP" install colorlog einops --quiet
"$PIP" install "hydra-core>=1.3" omegaconf --quiet
"$PIP" install hydra-zen --quiet
"$PIP" install "lightning>=2.0" pytorch-lightning --quiet
"$PIP" install "timm>=0.9" --quiet
"$PIP" install --no-build-isolation smplx chumpy --quiet
"$PIP" install wis3d --quiet

# 10. pytorch3d build prerequisites
echo "  [10/10] fvcore, iopath, ninja ..."
"$PIP" install fvcore iopath ninja --quiet

success "All pip dependencies installed"

# =============================================================================
# STEP 4 — pytorch3d (source build only — prebuilt wheels not used)
# =============================================================================
info "Step 4 — pytorch3d (source build)"

# 'import pytorch3d' alone is NOT a reliable health check: pytorch3d/__init__.py
# does not eagerly import the native _C extension, so a build/wheel with a
# broken _C.so (wrong ABI, missing CUDA runtime libs on a CPU box, etc.) will
# still pass a plain import — the failure only surfaces later when something
# calls into pytorch3d.ops. We force that exercise here so a bad install is
# caught and rebuilt now, not mid-pipeline in lift_gvhmr.py.
P3D_HEALTH='import pytorch3d; import pytorch3d.ops.knn'

if "$PYTHON3" -c "$P3D_HEALTH" 2>/dev/null; then
    P3D_VER=$("$PYTHON3" -c "import pytorch3d; print(pytorch3d.__version__)")
    success "pytorch3d $P3D_VER already installed and _C loads OK — skipping"
else
    # Prebuilt wheels intentionally NOT attempted. The PyTorch CDN matrix is
    # sparse for newer torch/Python combos, the community indexes have shipped
    # _C.so files that fail to load on this exact host+driver, and source build
    # against the live venv torch is the only consistently reliable option.

    if [[ "$GPU_MODE" == "cuda" && $HAS_NVCC -eq 0 ]]; then
        die "pytorch3d source build cannot proceed: nvcc not on PATH.

You're running CUDA torch but the CUDA toolkit (compiler) is missing. This
almost always means you picked a *runtime* container image instead of a
*devel* one. The runtime image has libcudart for executing CUDA code but
not nvcc for compiling it.

Fix on vast.ai (recommended):
  Stop and recreate the instance with this Docker image:
    pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
  (note: -devel, not -runtime). Then re-run this script.

Fix on a bare-metal box:
  $SUDO apt install -y nvidia-cuda-toolkit
  Then re-run this script. (Heads up: this installs a system CUDA that may
  not match your driver's CUDA version — prefer the devel image if you can.)"
    fi

    echo ""
    if [[ "$GPU_MODE" == "cuda" ]]; then
        echo "  Building pytorch3d from source with CUDA support. 5-10 min, do not interrupt."
    else
        echo "  Building pytorch3d from source (CPU only). 15-25 min, do not interrupt."
    fi
    echo ""

    # FORCE_CUDA pinned unconditionally: default 0, overwrite to 1 only in
    # CUDA mode. Without this, setup.py probes for a CUDA toolkit and will
    # happily compile CUDA kernels into _C.so even on a CPU-only torch venv,
    # producing a _C.so that fails to load at runtime (missing libcudart)
    # rather than failing to build.
    export MAX_JOBS=4
    export FORCE_CUDA=0
    if [[ "$GPU_MODE" == "cuda" ]]; then
        export FORCE_CUDA=1
    fi
    echo "  FORCE_CUDA = $FORCE_CUDA"
    [[ -n "${CUDA_HOME:-}" ]] && echo "  CUDA_HOME  = $CUDA_HOME"

    # ninja was pip-installed in Step 3, but pip console scripts sometimes
    # land in .venv/bin without being invocable from the build subprocess.
    # Without ninja the build falls back to single-threaded distutils
    # compilation, taking 4-5x longer. Verify and reinstall if needed.
    if "$PYTHON3" -m ninja --version &>/dev/null; then
        NINJA_VER=$("$PYTHON3" -m ninja --version 2>/dev/null)
        success "ninja $NINJA_VER ready (build will use parallel compilation)"
    else
        warn "ninja not invocable via 'python -m ninja' — reinstalling"
        "$PIP" install --force-reinstall ninja --quiet
        if "$PYTHON3" -m ninja --version &>/dev/null; then
            success "ninja reinstalled and now invocable"
        else
            warn "ninja still not invocable — build will use slower distutils compilation"
        fi
    fi

    # Build against HEAD of pytorch3d main. We do not pin @stable because
    # that ref has lagged real torch releases before. HEAD against live torch
    # in the venv gets the freshest pytorch3d that knows about current torch.
    BUILD_EXIT=0
    "$PIP" install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git" || BUILD_EXIT=$?

    if [[ $BUILD_EXIT -eq 0 ]] && "$PYTHON3" -c "$P3D_HEALTH" 2>/dev/null; then
        P3D_VER=$("$PYTHON3" -c "import pytorch3d; print(pytorch3d.__version__)")
        success "pytorch3d $P3D_VER built and installed, _C loads OK"
    else
        die "pytorch3d source build failed.

Most common causes on vast.ai / cloud GPUs:
  1. nvcc missing — you picked a *runtime* CUDA image. Recreate with:
       pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
  2. CUDA_HOME points at the wrong toolkit. Check:
       echo \$CUDA_HOME
       nvcc --version
     The CUDA major version in nvcc must match the one PyTorch was built
     against (run 'python -c \"import torch; print(torch.version.cuda)\"').
  3. Out of memory during compile (small instances). Reduce parallelism:
       MAX_JOBS=2 ./setup_vast.sh
  4. Driver too old for the toolkit in the image. nvidia-smi must show a
     CUDA Version >= the image's CUDA.

Scroll up for the actual compiler error and fix accordingly. The build
failure shows the offending .cu/.cpp file and the underlying gcc/nvcc message."
    fi
fi

# =============================================================================
# STEP 5 — Import smoke test
# =============================================================================
info "Step 5 — Import smoke test"

"$PYTHON3" - <<'PYEOF'
import sys, importlib

REQUIRED = [
    ("fastapi",           "fastapi"),
    ("uvicorn",           "uvicorn[standard]"),
    ("multipart",         "python-multipart"),
    ("numpy",             "numpy"),
    ("cv2",               "opencv-python-headless"),
    ("scipy",             "scipy"),
    ("yaml",              "pyyaml"),
    ("easydict",          "easydict"),
    ("shapely",           "shapely"),
    ("matplotlib",        "matplotlib"),
    ("ultralytics",       "ultralytics"),
    ("transformers",      "transformers"),
    ("torch",             "torch"),
    ("torchvision",       "torchvision"),
    ("onnxruntime",       "onnxruntime / onnxruntime-gpu"),
    ("imageio",           "imageio[ffmpeg]"),
    ("imageio_ffmpeg",    "imageio-ffmpeg"),
    ("ffmpeg",            "ffmpeg-python"),
    ("smplx",             "smplx"),
    ("einops",            "einops"),
    ("hydra",             "hydra-core"),
    ("omegaconf",         "omegaconf"),
    ("hydra_zen",         "hydra-zen"),
    ("lightning",         "lightning"),
    ("pytorch_lightning", "pytorch-lightning"),
    ("timm",              "timm"),
    ("colorlog",          "colorlog"),
    ("wis3d",             "wis3d"),
    ("fvcore",            "fvcore"),
    ("iopath",            "iopath"),
    # pytorch3d is checked separately below — 'import pytorch3d' silently
    # passes even when _C.so fails to load, so we don't include it here.
]

OPTIONAL = [
    ("tkinter",  "python3-tk (apt install python3-tk)"),
    ("av",       "av (PyAV)"),
    ("chumpy",   "chumpy"),
]

failed = []
for mod, pip_name in REQUIRED:
    try:
        importlib.import_module(mod)
        print(f"  \033[32m✓\033[0m  {mod}")
    except ImportError as exc:
        print(f"  \033[31m✗\033[0m  {mod}  →  pip install {pip_name}  ({exc})")
        failed.append(pip_name)

for mod, pip_name in OPTIONAL:
    try:
        importlib.import_module(mod)
        print(f"  \033[32m✓\033[0m  {mod}  (optional)")
    except ImportError:
        print(f"  \033[33m⚠\033[0m  {mod}  (optional — not installed, OK)")

if failed:
    print(f"\n  \033[31mFailed: {', '.join(failed)}\033[0m")
    sys.exit(1)

# Dedicated pytorch3d health check (not in the generic loop above): a plain
# 'import pytorch3d' passes even when _C.so failed to load, because the
# package's __init__ does not eagerly import the native extension. The real
# test is whether one of the modules that actually USES _C imports cleanly,
# which is what lift_gvhmr.py does at runtime.
try:
    import pytorch3d
    import pytorch3d.ops.knn
    print(f"  \033[32m✓\033[0m  pytorch3d {pytorch3d.__version__} (_C loads OK)")
except Exception as exc:
    print(f"  \033[31m✗\033[0m  pytorch3d  →  _C extension failed to load ({exc})")
    print( "       Try: pip uninstall pytorch3d -y && \\")
    print( "            pip install --no-build-isolation \\")
    print( "              'git+https://github.com/facebookresearch/pytorch3d.git'")
    sys.exit(1)

print("\n  All required imports OK")

# Bonus: confirm runtime device + ONNX provider stack
try:
    import torch, onnxruntime as ort
    if torch.cuda.is_available():
        provs = ort.get_available_providers()
        cuda_ok = "CUDAExecutionProvider" in provs
        print(f"\n  \033[32m✓\033[0m  CUDA ready: torch={torch.cuda.get_device_name(0)}, "
              f"ORT CUDAExecutionProvider={'yes' if cuda_ok else 'NO — install onnxruntime-gpu'}")
    else:
        print(f"\n  \033[33m⚠\033[0m  Running in CPU mode")
except Exception as e:
    print(f"\n  \033[33m⚠\033[0m  runtime check skipped: {e}")
PYEOF

success "Smoke test passed"

# =============================================================================
# STEP 6 — Model file check
# =============================================================================
info "Step 6 — Model file check"

MISSING_FILES=0

check_file() {
    local path="$1" label="$2"
    if [[ -f "$path" ]]; then
        SIZE=$(du -sh "$path" 2>/dev/null | cut -f1)
        success "$label  ($SIZE)"
    else
        err "MISSING: $label"
        echo "         Expected at: $path"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
}

echo "  models/:"
check_file "$APP_DIR/models/player_detection_v26s.pt" "player_detection_v26s.pt"
check_file "$APP_DIR/models/vitpose-b-coco.onnx"      "vitpose-b-coco.onnx"
check_file "$APP_DIR/models/SV_kp"                    "SV_kp  (PnLCalib)"
check_file "$APP_DIR/models/SV_lines"                 "SV_lines  (PnLCalib)"

echo ""
echo "  GVHMR/inputs/checkpoints/:"
check_file "$CKPT_DIR/gvhmr/gvhmr_siga24_release.ckpt"     "gvhmr_siga24_release.ckpt"
check_file "$CKPT_DIR/hmr2/epoch=10-step=25000.ckpt"       "hmr2 epoch=10-step=25000.ckpt"
check_file "$CKPT_DIR/body_models/smplx/SMPLX_NEUTRAL.npz" "SMPLX_NEUTRAL.npz"
check_file "$CKPT_DIR/vitpose/vitpose-h-multi-coco.pth"    "vitpose-h-multi-coco.pth"
check_file "$CKPT_DIR/yolo/yolov8x.pt"                     "yolov8x.pt"

if [[ $MISSING_FILES -gt 0 ]]; then
    warn "$MISSING_FILES model file(s) missing — affected pipeline stages will fail at runtime."
    warn "Run  python download_models.py  to fetch the public ones."
    warn "GVHMR weights must be uploaded manually (license)."
fi

# =============================================================================
# STEP 7 — Runtime directories
# =============================================================================
info "Step 7 — Runtime directories"
mkdir -p "$APP_DIR/uploads" "$APP_DIR/outputs" "$APP_DIR/outputs/transcode"
success "uploads/, outputs/, outputs/transcode/ ready"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  Setup complete!${RESET}"
echo ""
echo "  Start the server:"
echo ""
echo "    source .venv/bin/activate"
echo "    python main.py"
echo ""
if [[ "$GPU_MODE" == "cuda" ]]; then
    echo "  GPU mode: CUDA ($CUDA_VER, wheels $CUDA_TAG)"
else
    echo "  GPU mode: CPU (slow — expect 5-10x real time for the pipeline)"
fi
echo ""
echo "  On vast.ai / RunPod / Lambda:"
echo "    The server binds 0.0.0.0:8000 inside the container."
echo "    Your dashboard URL is the EXTERNAL port mapped to :8000 — find it"
echo "    in the vast 'Instances' panel (looks like host123.vast.ai:34567)."
echo ""
echo "  Tip: run inside tmux so disconnects don't kill the server:"
echo "    tmux new -s forma  →  source .venv/bin/activate && python main.py"
echo "    Detach with Ctrl-b d ; reattach with  tmux attach -t forma"
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo ""
