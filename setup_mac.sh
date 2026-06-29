#!/usr/bin/env bash
# =============================================================================
# FORMA 3D — macOS / Apple Silicon (M1/M2/M3) Setup Script
#
# Place this file inside the app/ folder, then run:
#
#   cd app
#   chmod +x setup_mac.sh
#   ./setup_mac.sh
#
# Requirements:
#   - macOS 12+, Apple Silicon (M1/M2/M3)
#   - Python 3.10 or 3.11  (brew install python@3.11  OR  python.org installer)
#   - Xcode Command Line Tools  (xcode-select --install)
#   - Internet access for pip + git
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
echo -e "${BOLD}  FORMA 3D — macOS Setup (Apple Silicon)               ${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo ""

# ── Paths (everything is relative to app/ where this script lives) ────────────
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/.venv"
CKPT_DIR="$APP_DIR/GVHMR/inputs/checkpoints"

# =============================================================================
# STEP 0 — Preflight checks
# =============================================================================
info "Step 0 — Preflight checks"

# Architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    success "Architecture: arm64 (Apple Silicon) — MPS will be used"
else
    warn "Architecture: $ARCH (Intel Mac) — MPS unavailable, pipeline runs on CPU"
fi

# Xcode CLI tools
if ! xcode-select -p &>/dev/null; then
    die "Xcode Command Line Tools not found.\nRun:  xcode-select --install\nThen re-run this script."
fi
success "Xcode CLI tools: $(xcode-select -p)"

# Python — prefer 3.11, fall back to 3.10, then generic python3
PYTHON=""
for candidate in python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$(command -v "$candidate")"
        break
    fi
done
[[ -z "$PYTHON" ]] && die "Python 3.10+ not found.\nInstall:  brew install python@3.11  OR  https://python.org"

PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

[[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]] && \
    die "Python $PY_VER found but 3.10+ is required."
[[ "$PY_MINOR" -ge 12 ]] && \
    warn "Python $PY_VER — 3.12+ is untested with GVHMR. 3.11 is safest."

success "Python: $PYTHON ($PY_VER)"

# git (needed for pytorch3d source build)
command -v git &>/dev/null || die "git not found. Run:  xcode-select --install"
success "git: $(git --version)"

# Homebrew — required for python-tk
if ! command -v brew &>/dev/null; then
    die "Homebrew not found. Install it first:\n  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"\nThen re-run this script."
fi
success "Homebrew: $(brew --version | head -1)"

# ── Tkinter / python-tk fix ───────────────────────────────────────────────────
# GVHMR imports turtle/tkinter internally. Homebrew Python ships without the
# Tcl/Tk bindings by default, causing "import _tkinter" errors at runtime.
# We install python-tk@<version> and set PKG_CONFIG_PATH so the bindings
# are found both by the system Python and the venv built on top of it.
info "Step 0b — Tkinter / Tcl-Tk fix"

TK_FORMULA="python-tk@${PY_VER}"
echo "  Installing ${TK_FORMULA} via Homebrew ..."
brew install "$TK_FORMULA" 2>/dev/null || {
    warn "${TK_FORMULA} not found in Homebrew, trying python-tk ..."
    brew install python-tk 2>/dev/null || warn "python-tk install failed — tkinter may be missing at runtime"
}

# Set PKG_CONFIG_PATH so tcl-tk headers/libs are discoverable during any
# source builds that compile against Tcl/Tk, and exported so child processes inherit it.
TCL_TK_PREFIX="$(brew --prefix tcl-tk@8 2>/dev/null || brew --prefix tcl-tk 2>/dev/null || echo "")"
if [[ -n "$TCL_TK_PREFIX" ]]; then
    export PKG_CONFIG_PATH="${TCL_TK_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    success "PKG_CONFIG_PATH set → ${TCL_TK_PREFIX}/lib/pkgconfig"
else
    warn "tcl-tk prefix not found via brew — PKG_CONFIG_PATH not set (tkinter may still work)"
fi

# Verify tkinter is importable with the chosen Python
if "$PYTHON" -c "import _tkinter" 2>/dev/null; then
    success "tkinter (_tkinter) importable with $PYTHON"
else
    warn "tkinter not yet importable with system Python — will be checked again after venv is created"
fi

# =============================================================================
# STEP 1 — Virtual environment
# =============================================================================
info "Step 1 — Virtual environment"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "  Creating .venv inside app/ ..."
    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created"
else
    success "Virtual environment already exists — reusing"
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
    warn "tkinter not importable inside venv. If GVHMR fails with '_tkinter' errors, run:\n  brew install ${TK_FORMULA}\n  then recreate the venv: rm -rf .venv && ./setup_mac.sh"
fi

# =============================================================================
# STEP 2 — PyTorch with MPS support (native arm64 wheel)
# =============================================================================
info "Step 2 — PyTorch with MPS support"

if "$PYTHON3" -c "
import torch, sys
major, minor = (int(x) for x in torch.__version__.split('.')[:2])
sys.exit(0 if (major, minor) >= (2, 1) else 1)
" 2>/dev/null; then
    TORCH_VER=$("$PYTHON3" -c "import torch; print(torch.__version__)")
    success "PyTorch $TORCH_VER already installed — skipping"
else
    echo "  Installing PyTorch + torchvision ..."
    # Plain PyPI wheel on arm64 macOS ships with Metal/MPS built in.
    # Do NOT add --index-url here — the CUDA index has no arm64 wheels.
    "$PIP" install torch torchvision --quiet
    success "PyTorch installed"
fi

"$PYTHON3" - <<'PYEOF'
import torch
mps = getattr(torch.backends, "mps", None)
if mps and mps.is_available():
    print(f"  \033[32m✓\033[0m  MPS (Metal GPU) available — torch {torch.__version__}")
else:
    print(f"  \033[33m⚠\033[0m  MPS not available — CPU fallback will be used (torch {torch.__version__})")
PYEOF

# =============================================================================
# STEP 3 — All pip dependencies
# =============================================================================
info "Step 3 — pip dependencies"

# 1. numpy first — pinned to 1.26.4 exactly.
# onnxruntime is compiled against numpy 1.x ABI; numpy 2.x causes a segfault
# inside onnxruntime_pybind11_state.so at CreateTensor (confirmed crash on M1).
echo "  [1/10] numpy, scipy, shapely, pyyaml, easydict, matplotlib ..."
"$PIP" install "numpy==1.26.4" scipy shapely pyyaml easydict matplotlib --quiet

# 2. FastAPI server
echo "  [2/10] fastapi, uvicorn, python-multipart ..."
"$PIP" install fastapi "uvicorn[standard]" python-multipart --quiet

# 3. OpenCV
echo "  [3/10] opencv-python ..."
"$PIP" install opencv-python --quiet

# 4. Video I/O
echo "  [4/10] imageio, imageio-ffmpeg, av ..."
"$PIP" install "imageio[ffmpeg]" imageio-ffmpeg --quiet
# av is optional — imageio has a built-in fallback if av is absent
"$PIP" install av --quiet \
    || warn "av (PyAV) failed — imageio fallback reader will be used (OK)"

# 5. ffmpeg-python
# GVHMR does  import ffmpeg  which maps to this package.
# The unrelated bare 'ffmpeg' stub on PyPI must NOT be installed.
echo "  [5/10] ffmpeg-python ..."
"$PIP" install ffmpeg-python --quiet

# 6. YOLO + BoT-SORT tracking
echo "  [6/10] ultralytics ..."
"$PIP" install ultralytics --quiet

# 7. Team clustering (ResNet-26 embeddings)
echo "  [7/10] transformers ..."
"$PIP" install transformers --quiet

# 8. ONNX Runtime — installed AFTER numpy==1.26.4 is locked in.
# onnxruntime must link against numpy 1.x ABI. Installing it before numpy
# is pinned, or letting pip resolve it first, causes a segfault at runtime
# (EXC_BAD_ACCESS in CreateTensor on macOS arm64).
echo "  [8/10] onnxruntime ..."
if [[ "$ARCH" == "arm64" ]]; then
    "$PIP" install onnxruntime-silicon --quiet 2>/dev/null \
        || "$PIP" install onnxruntime --quiet \
        || warn "onnxruntime install failed — ViTPose will error at runtime"
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
# STEP 4 — pytorch3d (build from source, ~10-20 min)
# =============================================================================
info "Step 4 — pytorch3d (source build)"
echo ""
echo "  No pre-built arm64 wheel exists on PyPI — compiling from source."
echo "  This takes 10-20 min on M1. Do not interrupt."
echo ""

if "$PYTHON3" -c "import pytorch3d" 2>/dev/null; then
    P3D_VER=$("$PYTHON3" -c "import pytorch3d; print(pytorch3d.__version__)")
    success "pytorch3d $P3D_VER already installed — skipping build"
else
    # MAX_JOBS=4 prevents OOM on 16 GB machines during parallel compile
    export MAX_JOBS=4
    # Suppress clang deprecation noise on newer macOS SDKs
    export MACOSX_DEPLOYMENT_TARGET=12.0
    # No CUDA on Apple Silicon — disable CUDA extension detection
    export FORCE_CUDA=0

    "$PIP" install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

    if "$PYTHON3" -c "import pytorch3d" 2>/dev/null; then
        P3D_VER=$("$PYTHON3" -c "import pytorch3d; print(pytorch3d.__version__)")
        success "pytorch3d $P3D_VER built and installed"
    else
        die "pytorch3d compiled but import failed. Check build output above."
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
    ("cv2",               "opencv-python"),
    ("scipy",             "scipy"),
    ("yaml",              "pyyaml"),
    ("easydict",          "easydict"),
    ("shapely",           "shapely"),
    ("matplotlib",        "matplotlib"),
    ("ultralytics",       "ultralytics"),
    ("transformers",      "transformers"),
    ("torch",             "torch"),
    ("torchvision",       "torchvision"),
    ("onnxruntime",       "onnxruntime / onnxruntime-silicon"),
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
    ("pytorch3d",         "pytorch3d (source build)"),
]

OPTIONAL = [
    ("tkinter",  "python-tk (brew install python-tk@3.10 or @3.11)"),
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

print("\n  All required imports OK")
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
echo "  To start the server:"
echo ""
echo "    source .venv/bin/activate"
echo "    python main.py"
echo ""
echo "  Then open:  http://localhost:8000"
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo ""