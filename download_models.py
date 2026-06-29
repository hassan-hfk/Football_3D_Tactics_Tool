#!/usr/bin/env python3
"""
download_models.py — fetch all model weights needed by the Football 3D Tactics pipeline.

Run once after cloning:
    python download_models.py

Models downloaded:
  - YOLOv8 player detection    (player_detection_v26s.pt)
  - ViTPose-B COCO skeleton    (vitpose-b-coco.onnx)
  - PnLCalib keypoint model    (models/SV_kp)
  - PnLCalib line model        (models/SV_lines)

For GVHMR weights, follow the instructions in README.md — they require
a manual download from the original repo due to license restrictions.
"""

import os
import urllib.request
import zipfile

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def download(url, dest, desc=""):
    if os.path.exists(dest):
        print(f"  already exists, skipping: {os.path.basename(dest)}")
        return
    print(f"  downloading {desc or os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  done: {dest}")


# ── YOLOv8 player detection ───────────────────────────────────────────────────
# Replace URL with your actual hosted weight URL (Google Drive, HuggingFace, etc.)
# Example using HuggingFace:
# download(
#     "https://huggingface.co/YOUR_USERNAME/football-tactics/resolve/main/player_detection_v26s.pt",
#     os.path.join(MODELS_DIR, "player_detection_v26s.pt"),
#     "YOLOv8 player detection",
# )

# ── ViTPose-B ONNX ────────────────────────────────────────────────────────────
# download(
#     "https://huggingface.co/YOUR_USERNAME/football-tactics/resolve/main/vitpose-b-coco.onnx",
#     os.path.join(MODELS_DIR, "vitpose-b-coco.onnx"),
#     "ViTPose-B COCO",
# )

# ── PnLCalib weights ──────────────────────────────────────────────────────────
# PnLCalib ships its own weights separately. Clone the repo and follow:
# https://github.com/mguti97/PnLCalib
# Then copy SV_kp and SV_lines into app/models/

print("Model download script — add your hosted URLs above and re-run.")
print("See README.md for manual download instructions for GVHMR weights.")
