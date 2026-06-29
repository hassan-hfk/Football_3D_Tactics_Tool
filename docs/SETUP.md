# FORMA 3D — Setup Guide

Complete installation instructions for new users on **Windows**, **Linux**, and **macOS**.

---

## Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.10 or 3.11 | 3.12 not tested with GVHMR |
| CUDA (optional) | 11.8+ | For GPU acceleration on NVIDIA hardware |
| RAM | 16 GB | 32 GB recommended for full-clip GVHMR runs |
| Disk | 10 GB free | Models + checkpoints + outputs |
| Git | any | For cloning the GVHMR repo |

---

## Step 1 — Clone / open the project

The project folder is `Football_pitch_keypoints_training/app/`. All commands below are run from inside the `app/` directory unless noted otherwise.

```bash
cd Football_pitch_keypoints_training/app
```

---

## Step 2 — Create a virtual environment

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## Step 3 — Install PyTorch

PyTorch must be installed **before** the rest of the requirements because the correct variant depends on your hardware.

### NVIDIA GPU (CUDA 11.8)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### NVIDIA GPU (CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### CPU only (or Apple Silicon — MPS is included automatically)
```bash
pip install torch torchvision
```

Verify your install:
```python
import torch
print(torch.__version__)          # e.g. 2.3.0
print(torch.cuda.is_available())  # True if NVIDIA GPU detected
```

---

## Step 4 — Install all other dependencies

```bash
pip install -r requirements.txt
```

This installs: FastAPI, YOLO (Ultralytics), PnLCalib deps, ViTPose ONNX runtime, GVHMR deps (einops, hydra-core, lightning, timm, colorlog, hydra-zen, wis3d), SMPL-X, imageio with ffmpeg, imageio-ffmpeg (bundled ffmpeg binary for MP4 export), and more. See `requirements.txt` for the full list.

> **Windows note:** `av` (PyAV) sometimes needs the Visual C++ redistributable. If it fails, try `pip install av --no-binary av` or simply skip it — `imageio` will fall back to a slower reader.

---

## Step 5 — pytorch3d

pytorch3d is required by `lift_gvhmr.py` for rotation math (axis-angle ↔ matrix conversions). It is **not** on PyPI for most setups and must be built from source.

### Option A — Pre-built wheel (Windows / Linux, easiest)

Check the [pytorch3d releases page](https://github.com/facebookresearch/pytorch3d/releases) for a wheel matching your Python and PyTorch version. If one exists:

```bash
pip install <downloaded_wheel>.whl
```

### Option B — Build from source (all platforms)

```bash
pip install fvcore iopath ninja
# Set MAX_JOBS to avoid running out of RAM during compilation
# Windows PowerShell:
$env:MAX_JOBS = "4"
# macOS / Linux:
export MAX_JOBS=4

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

This takes 10–30 minutes depending on your machine. You need a C++ compiler:
- **Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with the "Desktop development with C++" workload
- **Linux:** `sudo apt install build-essential`
- **macOS:** `xcode-select --install`

Verify:
```python
import pytorch3d
print(pytorch3d.__version__)
```

---

## Step 6 — Place model files

All model files go in `app/models/`. The directory should look like:

```
models/
  player_detection_v26s.pt   # 4-class YOLO detector (Ball, Goalkeeper, Player, Referee)
  pitch_keypoints.pt         # Legacy 29-keypoint pitch model (yolo backend only)
  SV_kp/                     # PnLCalib single-view keypoint weights
  SV_lines/                  # PnLCalib single-view line weights
  vitpose-b-coco.onnx        # ViTPose-B body pose (ONNX export)
```

If `SV_kp` / `SV_lines` are missing, download the PnLCalib single-view weights from the [PnLCalib repository](https://github.com/mguti97/No-Bells-Just-Whistles) and place them at those paths.

---

## Step 7 — GVHMR checkpoints (for mesh generation)

GVHMR mesh lifting (`lift_gvhmr.py`) requires model checkpoints that must be downloaded separately from HuggingFace.

The expected directory layout inside `research/GVHMR/` is:

```
research/GVHMR/
  inputs/
    checkpoints/
      gvhmr/
        gvhmr_siga24_release.ckpt   # Main GVHMR checkpoint
      hmr2/
        epoch=10-step=25000.ckpt    # HMR2 ViT backbone
      body_models/
        smplx/
          SMPLX_NEUTRAL.npz         # SMPL-X neutral body model
```

### Download GVHMR checkpoints

```bash
# From the project root (one level above app/)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='zju3dv/GVHMR',
    local_dir='research/GVHMR/inputs/checkpoints',
    ignore_patterns=['*.md']
)
"
```

Alternatively, download manually from https://huggingface.co/zju3dv/GVHMR

### Download SMPL-X body model

1. Register at https://smpl-x.is.tue.mpg.de (free)
2. Download **SMPL-X v1.1** (the `.npz` version)
3. Place `SMPLX_NEUTRAL.npz` at:
   ```
   research/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz
   ```

### Clone the GVHMR repository

If `research/GVHMR/` is not already present:

```bash
# From the project root
mkdir -p research
git clone https://github.com/zju3dv/GVHMR.git research/GVHMR
```

---

## Step 8 — Verify the GVHMR setup

Test that the mesh pipeline can at least load models:

```bash
cd app
python lift_gvhmr.py outputs/<any_job_id>.json --max-tracks 1 --verts fp16
```

You should see lines like:
```
[lift_gvhmr] device      = cuda
[lift_gvhmr] GVHMR  : .../research/GVHMR
  GVHMR loaded   : gvhmr_siga24_release.ckpt on cuda
  HMR2 loaded    : epoch=10-step=25000.ckpt on cuda
```

If you get `FileNotFoundError` on a checkpoint, re-check Step 7.

---

## Step 9 — Run the server

```bash
cd app
python main.py
```

Open **http://localhost:8000** in your browser.

Expected startup output:
```
[server] Starting Football 3D Tactics server (Forma B)…
[server] Open http://localhost:8000 in your browser
[server] ffmpeg: C:\...\imageio_ffmpeg\binaries\ffmpeg-...exe
```

The last line confirms the bundled ffmpeg was found. If it says "ffmpeg not found", run:
```bash
pip install imageio-ffmpeg
```

---

## Typical First Run

1. Click **▲ Upload Video** and choose a broadcast football clip (MP4, AVI, MOV, MKV)
2. The pipeline runs automatically — watch the progress overlay:
   - "Loading models…" (first run only, slower due to model caching)
   - "Frame X / Y" (detection + tracking + homography per frame)
3. When done, the 3D tactical view loads. Players appear on the pitch with team colours.
4. Click **⬡ Generate Meshes** to run GVHMR mesh lifting (requires checkpoints from Step 7)
5. Watch the overlay: "Loading models…" → "Processing track N (player, M frames)" → done
6. The MESH tab loads with SMPL-X bodies on the pitch

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'colorlog'`
```bash
pip install colorlog
```

### `ModuleNotFoundError: No module named 'einops'`
```bash
pip install einops
```

### `ModuleNotFoundError: No module named 'hydra_zen'`
```bash
pip install hydra-zen
```

### `ModuleNotFoundError: No module named 'pytorch_lightning'`
```bash
pip install pytorch-lightning lightning
```

### `ModuleNotFoundError: No module named 'timm'`
```bash
pip install timm
```

### `ModuleNotFoundError: No module named 'wis3d'`
```bash
pip install wis3d
```

### `ModuleNotFoundError: No module named 'imageio'`
```bash
pip install imageio[ffmpeg] imageio-ffmpeg
```

### `ModuleNotFoundError: No module named 'ffmpeg'` (from GVHMR)
```bash
pip install ffmpeg-python
```
> Note: do **not** `pip install ffmpeg` — that is a different unrelated package. The correct one is `ffmpeg-python`.

### ffmpeg not found at `/transcode-to-mp4`
```bash
pip install imageio-ffmpeg
```
The `imageio-ffmpeg` package ships a self-contained ffmpeg binary that the server finds automatically.

### `FileNotFoundError: GVHMR checkpoint not found`
Re-read Step 7. Make sure both the GVHMR `.ckpt` files and `SMPLX_NEUTRAL.npz` are in the correct locations.

### `GlobalHydraAlreadyInitialized` error
This is handled automatically by `lift_gvhmr.py`. If it appears, it means Hydra was left in a broken state by a previous failed import. Restart the server.

### CUDA out of memory during GVHMR
Reduce the batch size in `lift_gvhmr.py`:
```python
HMR2_BATCH_SIZE_CUDA = 8   # default 16 — lower if OOM
```
Or limit the number of tracks:
```bash
python lift_gvhmr.py outputs/<id>.json --max-tracks 6
```

### Homography keeps returning "NO" for most frames
Loosen the quality gates in `pnl_homography.py`:
```python
PNL_MAX_REP_ERR_PX = 25.0   # default 18.0
PNL_MIN_KEYPOINTS  = 4      # default 6
```

### Windows: `av` fails to install
Skip it — `imageio` will fall back to a slower reader automatically. The pipeline still works.

### Windows: PowerShell execution policy blocks `.venv\Scripts\Activate.ps1`
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then rerun the activate command.

---

## Environment Summary

After a successful setup, `pip list` should include at minimum:

```
fastapi, uvicorn, python-multipart
numpy, opencv-python, scipy, shapely
ultralytics
torch, torchvision
onnxruntime
smplx, einops, hydra-core, omegaconf, lightning, timm
colorlog, hydra-zen, wis3d
imageio, av, imageio-ffmpeg, ffmpeg-python
pytorch3d
```
