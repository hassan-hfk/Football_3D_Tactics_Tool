"""
lift_gvhmr.py — add SMPL-X body meshes to a processed result JSON.

Pairs with pipeline.py the same way lift_motionbert.py does:
    pipeline.py  -> outputs/<id>.json          (per-frame players, bboxes, kpts)
    lift_motionbert.py outputs/<id>.json -> outputs/<id>_3d.json   (skeleton)
    lift_gvhmr.py     outputs/<id>.json -> outputs/<id>_mesh.json  (SMPL-X mesh)

The vendored GVHMR repo (https://github.com/zju3dv/GVHMR, SIGGRAPH Asia 2024)
predicts SMPL-X parameters per frame for a tracked person. We reuse:
  - HMR2.0a ViT backbone        — image features per crop
  - GVHMR transformer + denoiser — SMPL-X params from features + ViTPose

We DON'T reuse:
  - YoloV8 tracking      (pipeline.py already runs BoT-SORT with stable IDs)
  - ViTPose-H            (pose.py already runs ViTPose-B; format is identical)
  - SimpleVO / DPVO      (broken on flat football pitch; we use static_cam=True
                          and place meshes on the pitch via PnLCalib pitch_pos
                          downstream in the viewer)

Per-track flow:
  1. Group player entries by track id from result.json.
  2. Trim each track to its longest contiguous segment (filling small gaps
     with linear interpolation, exactly like GVHMR's tracker.get_one_track).
  3. Read the video frames belonging to that segment.
  4. Run HMR2 ViT to get (F, 1024) image features.
  5. Pack {kp2d, bbx_xys, K, cam_angvel=zeros, f_imgseq} and call GVHMR's
     pipeline.forward(..., static_cam=True). Get camera-frame SMPL-X.
  6. Store SMPL-X params per frame, keyed by track id.

Output additions vs result.json:
  player["smplx"] = {
    "body_pose":     63 floats,   # axis-angle, 21 body joints (camera frame)
    "global_orient": 3 floats,    # axis-angle, root           (camera frame)
    "betas":         10 floats,   # shape
    "transl":        3 floats,    # camera-frame translation (metres)
    "gvhmr_yaw":     1 float,     # radians; the yaw component of GVHMR's
                                  # global-frame global_orient BEFORE we
                                  # zero it for the verts forward. Viewer
                                  # uses this to track real body rotation
                                  # within a track and detect backpedaling
                                  # (velocity ≠ body facing). Only present
                                  # when --verts fp16|fp32.
    "verts_b64":     str,         # base64 float16/32 (F, 10475, 3), present
                                  # only when --verts fp16|fp32. Vertices are
                                  # rendered with yaw stripped (body faces a
                                  # canonical direction); the viewer applies
                                  # mesh.rotation.y = gvhmr_yaw + per-track
                                  # calibration constant so the body's real
                                  # facing direction shows up on the pitch.
                                  # Floor-translated so the lowest vertex
                                  # sits at Y=0 — plant the mesh at pitch_pos
                                  # with Y=0 and feet rest on the grass.
  }
  metadata["smplx"] = {
    "model"            : "smplx",
    "num_betas"        : 10,
    "num_body_pose_dim": 63,
    "convention"       : "axis_angle",
    "params_frame"     : "camera",          # body_pose / global_orient / transl
    "verts_frame"      : "canonical_yup",   # verts_b64: yaw=0, Y-up, floor=0
    "verts_yaw"        : "zeroed",          # viewer must apply heading
    "yaw_source"       : "gvhmr_per_frame", # per-frame gvhmr_yaw is emitted
    "K_fullimg"        : [[fx,0,cx],[0,fy,cy],[0,0,1]],   # what we passed to GVHMR
  }

Device portability:
  Uses _pick_device() (cuda > mps > cpu). We override GVHMR's hardcoded .cuda()
  calls with .to(device) by subclassing the runner. Note: GVHMR's source
  imports pytorch3d at module load (rotation utilities + ops.knn). pytorch3d
  itself runs CPU-fine for rotation maths, but its CUDA-only build chain
  means MPS / CPU users may need the source build to succeed without GPU
  kernels. See app/README_GVHMR.md for setup specifics.

RUN:
  python lift_gvhmr.py outputs/af07ed52.json
  -> writes outputs/af07ed52_mesh.json

  python lift_gvhmr.py outputs/af07ed52.json --device cpu
  -> force CPU regardless of available GPU

  python lift_gvhmr.py outputs/af07ed52.json --video uploads/af07ed52_clip.mp4
  -> use a specific video file (default: auto-discover from result.json)
"""

import os
import sys
import json
import time
import base64
import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Config — paths and tunables. Edit here, not in code below.
# ─────────────────────────────────────────────────────────────────────────────

# Where the GVHMR repo lives. Set via the GVHMR_DIR env var, or default to
# either of these two project-relative locations:
GVHMR_DIR_DEFAULT_CANDIDATES = [
    "../research/GVHMR",         # sibling of app/, matches Hassan's layout
    "../GVHMR",                  # alternative
    "research/GVHMR",            # if run from project root
    "GVHMR",
]

# Checkpoint paths inside GVHMR_DIR (these are the standard GVHMR locations,
# downloaded into research/GVHMR/inputs/checkpoints — see README_GVHMR.md).
GVHMR_CKPT_REL   = "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
HMR2_CKPT_REL    = "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
BODY_MODELS_REL  = "inputs/checkpoints/body_models"  # holds smplx/SMPLX_NEUTRAL.npz

# Device for GVHMR / HMR2 inference. "auto" picks cuda > mps > cpu.
DEVICE_PREF = "auto"

# Per-track filters. Tracks shorter than MIN_TRACK_LEN frames or with median
# bbox height < MIN_BBOX_H_PX are dropped — GVHMR has no anchor on tiny
# crops and produces unstable meshes that look worse than no mesh at all.
MIN_TRACK_LEN  = 8
MIN_BBOX_H_PX  = 10

# Ball tracks don't get meshes. Players, goalkeepers AND referees do, so the
# mesh page mirrors the tactical page: referees show as referees (their own
# colour) instead of being folded into a team or dropped from the scene.
KIND_FILTER = {"player", "goalkeeper", "referee"}

# Maximum gap (in PROCESSED frames) we'll bridge with linear interpolation
# inside a single track segment. Larger gaps split the track into separate
# segments processed independently.
MAX_INTERP_GAP = 6

# How many frames per HMR2 batch. 16 fits comfortably on a T4 (16 GB) and
# 4 fits on most CPU/MPS setups.
HMR2_BATCH_SIZE_CUDA = 16
HMR2_BATCH_SIZE_CPU  = 4

# Mesh rendering output. When enabled, lift_gvhmr.py runs an SMPL-X forward
# pass per frame and embeds the 10475 vertex coordinates as base64 in the
# JSON, so a browser viewer can render the mesh without doing SMPL-X math in
# JavaScript. fp16 halves the file size at imperceptible visual cost.
#   "off"  - SMPL-X parameters only (small JSON, browser must do forward)
#   "fp16" - parameters + base64 vertices as float16 (recommended)
#   "fp32" - parameters + base64 vertices as float32 (matches Kaggle format)
VERTS_DTYPE_DEFAULT = "fp16"

# Default track cap. 0 means lift every track that passes the filters above.
# Use a small number (e.g. 6) for quick iteration on the viewer.
MAX_TRACKS_DEFAULT = 0

# Still-image input. GVHMR is a temporal transformer and cannot lift a single
# isolated frame, so an image is replicated into a short static clip of this
# many frames; the representative (middle) frame is written back to the one
# real frame. 16 gives the denoiser enough temporal context while staying
# cheap. See process()'s is_image branch and _build_image_segments().
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGE_REPLICATE_FRAMES = 16


# ─────────────────────────────────────────────────────────────────────────────
# Device picker — matches the pattern in pipeline.py and pnl_homography.py.
# ─────────────────────────────────────────────────────────────────────────────

def _pick_device(pref: str = DEVICE_PREF) -> str:
    import torch
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "cpu"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# GVHMR repo discovery & path setup. Must happen BEFORE we import hmr4d.*
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_gvhmr_dir() -> Path:
    """Find the GVHMR repo. Env var takes priority; otherwise try the defaults."""
    env = os.environ.get("GVHMR_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"GVHMR_DIR='{env}' is set but does not point to a directory."
            )
        return p

    here = Path(__file__).resolve().parent
    for rel in GVHMR_DIR_DEFAULT_CANDIDATES:
        p = (here / rel).resolve()
        if p.is_dir() and (p / "hmr4d").is_dir():
            return p
        # Also try relative to current working directory
        p_cwd = Path(rel).resolve()
        if p_cwd.is_dir() and (p_cwd / "hmr4d").is_dir():
            return p_cwd

    raise FileNotFoundError(
        "GVHMR repo not found. Set the GVHMR_DIR environment variable to the "
        "clone location, e.g. GVHMR_DIR=../research/GVHMR. See README_GVHMR.md."
    )


def _setup_gvhmr_imports(gvhmr_dir: Path) -> None:
    """Add the GVHMR repo to sys.path so `import hmr4d.*` resolves. Idempotent.

    Does NOT chdir — that's handled by `_chdir_to_gvhmr()` as a context
    manager because GVHMR's internal loaders use PROJ_ROOT-relative paths
    AND main.py / pipeline.py also use cwd-relative paths (uploads/, outputs/,
    static/). We must restore cwd when we're done.
    """
    sp = str(gvhmr_dir)
    if sp not in sys.path:
        sys.path.insert(0, sp)


@contextmanager
def _chdir_to_gvhmr(gvhmr_dir: Path):
    """Temporarily chdir to the GVHMR repo so its PROJ_ROOT-relative loaders
    work, then restore the original cwd. Use as `with _chdir_to_gvhmr(p): ...`.

    Restoration runs even if the block raises, including KeyboardInterrupt.
    """
    saved_cwd = os.getcwd()
    try:
        os.chdir(gvhmr_dir)
        yield
    finally:
        os.chdir(saved_cwd)


def _reset_hydra_if_initialized() -> None:
    """Hydra's GlobalHydra is a process-wide singleton. If anything in the
    process has called initialize/initialize_config_module before, the next
    call raises GlobalHydraAlreadyInitialized. We clear it defensively so
    process() can be called repeatedly (e.g. from a long-running FastAPI
    worker that handles multiple jobs)."""
    try:
        from hydra.core.global_hydra import GlobalHydra
        gh = GlobalHydra.instance()
        if gh.is_initialized():
            gh.clear()
    except Exception:
        # Hydra not importable yet — first call, nothing to clear.
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Track parsing — convert result.json into per-track sequences.
# ─────────────────────────────────────────────────────────────────────────────

def collect_tracks(result: dict) -> dict:
    """Group player entries from result.json by track id.

    Returns:
        {track_id: {
            "frames":   list of frame_idx (sorted, may have gaps),
            "bboxes":   list of [x1,y1,x2,y2] aligned to frames,
            "kpts":     list of (17, 3) lists aligned to frames; None where
                        ViTPose was skipped (tiny crops in pipeline.py),
            "kind":     "player" or "goalkeeper",
            "team":     0 / 1 / -1 / None,
        }}
    """
    tracks: dict = {}
    frames = result.get("frames", [])

    for f in frames:
        f_idx = f.get("frame_idx")
        if f_idx is None:
            continue
        for p in f.get("players", []):
            tid = p.get("id")
            if tid is None or tid < 0:
                continue
            kind = p.get("kind", "")
            if kind not in KIND_FILTER:
                continue
            tr = tracks.setdefault(tid, {
                "frames": [], "bboxes": [], "kpts": [],
                "kind": kind, "team": p.get("team"),
            })
            tr["frames"].append(int(f_idx))
            tr["bboxes"].append(p["bbox"])
            tr["kpts"].append(p.get("kpts"))
            # keep latest non-null team value
            if p.get("team") is not None:
                tr["team"] = p["team"]

    return tracks


def split_into_segments(track: dict, max_gap: int = MAX_INTERP_GAP) -> list:
    """Split one track into contiguous segments. A segment is allowed to
    contain gaps of at most max_gap PROCESSED frames; larger gaps split it.

    Returns:
        list of dicts with the same shape as a track but contiguous frames,
        plus a `valid_mask` list indicating which entries were originally
        present (True) versus interpolated (False).
    """
    frames = track["frames"]
    if not frames:
        return []

    # frames are already sorted by collect_tracks because we iterate result
    # frames in order, but enforce just in case.
    order = np.argsort(frames)
    frames  = [frames[i]  for i in order]
    bboxes  = [track["bboxes"][i]  for i in order]
    kpts    = [track["kpts"][i]    for i in order]

    segments = []
    cur_frames, cur_bboxes, cur_kpts = [frames[0]], [bboxes[0]], [kpts[0]]

    for i in range(1, len(frames)):
        gap = frames[i] - frames[i-1]
        if gap <= max_gap:
            # Same segment; interpolate intermediate frames if gap > 1.
            for g in range(1, gap):
                alpha = g / gap
                interp_bbox = [
                    bboxes[i-1][k] * (1 - alpha) + bboxes[i][k] * alpha
                    for k in range(4)
                ]
                cur_frames.append(frames[i-1] + g)
                cur_bboxes.append(interp_bbox)
                cur_kpts.append(None)        # mark as interpolated
            cur_frames.append(frames[i])
            cur_bboxes.append(bboxes[i])
            cur_kpts.append(kpts[i])
        else:
            # Start a new segment
            segments.append({
                "frames": cur_frames,
                "bboxes": cur_bboxes,
                "kpts":   cur_kpts,
                "kind":   track["kind"],
                "team":   track["team"],
            })
            cur_frames, cur_bboxes, cur_kpts = [frames[i]], [bboxes[i]], [kpts[i]]

    segments.append({
        "frames": cur_frames,
        "bboxes": cur_bboxes,
        "kpts":   cur_kpts,
        "kind":   track["kind"],
        "team":   track["team"],
    })
    return segments


def segment_passes_filters(segment: dict, min_len: int, min_bbox_h: float) -> bool:
    if len(segment["frames"]) < min_len:
        return False
    heights = [b[3] - b[1] for b in segment["bboxes"]]
    if np.median(heights) < min_bbox_h:
        return False
    return True


def _build_image_segments(tracks: dict, n_replicate: int) -> list:
    """Image-mode segment builder. Each track from a still image has exactly
    one real observation; GVHMR cannot lift a single frame, so we replicate it
    into an `n_replicate`-frame static clip (same bbox + kpts every frame).
    The frames are numbered 1..n_replicate, but only frame 1 exists in the
    result JSON — process() collapses the lifted clip back to it.

    The MIN_TRACK_LEN filter is met by construction (length == n_replicate),
    so only the bbox-height filter is applied here.
    """
    segs = []
    for tid, track in tracks.items():
        if not track["frames"]:
            continue
        bbox = track["bboxes"][0]
        kpt  = track["kpts"][0]
        if (bbox[3] - bbox[1]) < MIN_BBOX_H_PX:
            continue
        seg = {
            "frames": list(range(1, n_replicate + 1)),
            "bboxes": [bbox] * n_replicate,
            "kpts":   [kpt]  * n_replicate,   # static pose held on every frame
            "kind":   track["kind"],
            "team":   track["team"],
        }
        segs.append((tid, seg))
    return segs


# ─────────────────────────────────────────────────────────────────────────────
# Video frame cache — read once, serve by index.
# ─────────────────────────────────────────────────────────────────────────────

class FrameCache:
    """Reads a video (or a single still image) into memory at full resolution.
    Big videos use a lot of RAM but reading once per track instead of N times
    is essential for speed.

    Image mode: when the source is a still image, `length` is 1 and `slice()`
    returns that one image for every requested index, so the still can be
    replicated into the short static clip GVHMR needs (it is a temporal model
    and cannot lift a single isolated frame)."""

    def __init__(self, video_path: str, is_image: bool = False):
        # Import here so module import doesn't require imageio if the caller
        # never uses FrameCache (e.g. unit tests with synthetic inputs).
        import imageio.v3 as iio

        self.path = str(video_path)
        self.is_image = is_image

        if is_image:
            img = np.asarray(iio.imread(self.path))   # (H,W,3|4) or (H,W)
            if img.ndim == 2:                          # greyscale -> RGB
                img = np.stack([img] * 3, axis=-1)
            if img.shape[2] == 4:                      # drop alpha
                img = img[:, :, :3]
            self.image  = np.ascontiguousarray(img[:, :, :3]).astype(np.uint8)
            self.length = 1
            self.height = self.image.shape[0]
            self.width  = self.image.shape[1]
        else:
            # Read entire video as (T, H, W, 3) uint8 RGB
            self.frames = iio.imread(self.path, plugin="pyav")
            self.length = len(self.frames)
            self.height = self.frames.shape[1]
            self.width  = self.frames.shape[2]

    def slice(self, frame_indices: list) -> np.ndarray:
        """Return frames at 1-based indices as in pipeline.py's frame_idx.
        Indices are clipped to [0, length-1]; out-of-range becomes black.
        In image mode every requested index returns the single still."""
        arr = np.zeros((len(frame_indices), self.height, self.width, 3),
                       dtype=np.uint8)
        if self.is_image:
            arr[:] = self.image
            return arr
        for i, fi in enumerate(frame_indices):
            # pipeline.py uses 1-based raw_frame_num; convert to 0-based.
            j = fi - 1
            if 0 <= j < self.length:
                arr[i] = self.frames[j]
        return arr


# ─────────────────────────────────────────────────────────────────────────────
# GVHMRRunner — wraps the GVHMR model with device-aware inference.
#
# We don't subclass DemoPL because its predict() hard-codes .cuda(). We
# replicate the small predict() body with .to(device) instead. The model is
# instantiated by Hydra exactly as the demo does.
# ─────────────────────────────────────────────────────────────────────────────

class GVHMRRunner:
    def __init__(self, gvhmr_dir: Path, device: str):
        import torch
        from hydra import initialize_config_module, compose
        from hmr4d.configs import register_store_gvhmr
        from hmr4d.network.hmr2 import load_hmr2

        self.gvhmr_dir = Path(gvhmr_dir)
        self.device_str = device
        self.device = torch.device(device)

        # Compose the demo cfg (network + endecoder + model). We pass a
        # placeholder video_name because the schema requires it; we won't
        # use any of the path fields.
        _reset_hydra_if_initialized()
        with initialize_config_module(version_base="1.3",
                                      config_module="hmr4d.configs"):
            register_store_gvhmr()
            # Side-effect import: the `gvhmr_pl_demo` config is registered
            # at the bottom of this module file. store_gvhmr.py doesn't pull
            # it in (it only imports gvhmr_pl, the training variant), so the
            # demo cfg can't find it unless we import it here ourselves.
            import hmr4d.model.gvhmr.gvhmr_pl_demo  # noqa: F401
            self.cfg = compose(
                config_name="demo",
                overrides=["video_name=__lift_gvhmr_placeholder__",
                           "static_cam=True"],
            )

        # Instantiate the Lightning model (DemoPL) — _recursive_=False so we
        # control deep instantiation. This is exactly what demo.py does.
        import hydra
        self.model = hydra.utils.instantiate(self.cfg.model, _recursive_=False)
        ckpt_path = self.gvhmr_dir / GVHMR_CKPT_REL
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"GVHMR checkpoint not found at {ckpt_path}. "
                f"See README_GVHMR.md for the HuggingFace mirror."
            )
        self.model.load_pretrained_model(str(ckpt_path))
        self.model = self.model.eval().to(self.device)

        # HMR2 ViT feature extractor — also moved to our device.
        hmr2_ckpt = self.gvhmr_dir / HMR2_CKPT_REL
        if not hmr2_ckpt.exists():
            raise FileNotFoundError(
                f"HMR2 checkpoint not found at {hmr2_ckpt}. "
                f"See README_GVHMR.md."
            )
        self.hmr2 = load_hmr2(checkpoint_path=str(hmr2_ckpt))
        self.hmr2 = self.hmr2.eval().to(self.device)

        self.hmr2_batch = (HMR2_BATCH_SIZE_CUDA
                           if device == "cuda" else HMR2_BATCH_SIZE_CPU)

        print(f"  GVHMR loaded   : {ckpt_path.name} on {device}")
        print(f"  HMR2 loaded    : {hmr2_ckpt.name} on {device}")

    # ── HMR2 feature extraction ──────────────────────────────────────────
    @staticmethod
    def _preprocess_crops(images_np: np.ndarray, bbx_xys, img_dst_size: int = 256):
        """Crop and resize each frame around bbx_xys to (256, 256), normalize
        to HMR2's expected mean/std. Matches GVHMR's get_batch() with
        path_type='np', img_ds=1.0, exactly.

        Args:
            images_np: (F, H, W, 3) uint8 RGB
            bbx_xys:   torch tensor (F, 3) [cx, cy, size] in IMAGE pixels
        Returns:
            imgs:    torch float (F, 3, 256, 256), normalised
            bbx_xys: torch float (F, 3), unchanged (we ran at img_ds=1.0)
        """
        import torch
        from hmr4d.network.hmr2.utils.preproc import (
            crop_and_resize, IMAGE_MEAN, IMAGE_STD,
        )

        F = images_np.shape[0]
        gt_center   = bbx_xys[:, :2].numpy()
        gt_bbx_size = bbx_xys[:, 2].numpy()

        imgs_list = []
        for i in range(F):
            img, _bbx = crop_and_resize(
                images_np[i],
                gt_center[i],
                float(gt_bbx_size[i]),
                img_dst_size,
                enlarge_ratio=1.0,
            )
            imgs_list.append(img)
        imgs = np.stack(imgs_list)  # (F, 256, 256, 3) uint8
        imgs_t = torch.from_numpy(imgs).float() / 255.0
        imgs_t = (imgs_t - IMAGE_MEAN) / IMAGE_STD
        imgs_t = imgs_t.permute(0, 3, 1, 2).contiguous()  # (F, 3, 256, 256)
        return imgs_t, bbx_xys

    def extract_features(self, images_np: np.ndarray, bbx_xys):
        """Run HMR2 ViT on per-frame crops.

        Args:
            images_np: (F, H, W, 3) uint8 RGB
            bbx_xys:   torch.FloatTensor (F, 3)
        Returns:
            features:  torch.FloatTensor (F, 1024)
        """
        import torch
        imgs, _ = self._preprocess_crops(images_np, bbx_xys)
        imgs = imgs.to(self.device)
        features = []
        with torch.no_grad():
            for j in range(0, imgs.shape[0], self.hmr2_batch):
                batch = imgs[j:j + self.hmr2_batch]
                feat = self.hmr2({"img": batch})
                features.append(feat.detach().cpu())
        return torch.cat(features, dim=0).clone()  # (F, 1024)

    # ── GVHMR prediction ─────────────────────────────────────────────────
    def predict(self, data: dict, static_cam: bool = True) -> dict:
        """Device-aware copy of DemoPL.predict(). Returns SMPL-X params in
        BOTH camera frame (for downstream PnLCalib-based world placement) and
        global / gravity-aligned frame (for direct mesh rendering)."""
        import torch
        from hmr4d.utils.geo.hmr_cam import normalize_kp2d

        batch = {
            "length":    data["length"][None],
            "obs":       normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys":   data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "f_imgseq":  data["f_imgseq"][None],
        }
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = self.model.pipeline.forward(
                batch, train=False, postproc=True, static_cam=static_cam,
            )

        smpl_incam = {k: v[0].detach().cpu()
                      for k, v in outputs["pred_smpl_params_incam"].items()}
        smpl_global = {k: v[0].detach().cpu()
                       for k, v in outputs["pred_smpl_params_global"].items()}
        return {
            "smpl_params_incam":  smpl_incam,
            "smpl_params_global": smpl_global,
            "K_fullimg":          data["K_fullimg"].detach().cpu(),
        }

    # ── SMPL-X body model (loaded on demand) ─────────────────────────────
    def ensure_smplx(self):
        """Lazy-load the SMPL-X body model. Only loaded when verts are
        actually requested. Reads SMPLX_NEUTRAL.npz from GVHMR's body_models
        directory (the same file used by the GVHMR demo's renderer)."""
        if getattr(self, "_smplx_model", None) is not None:
            return self._smplx_model
        import torch
        try:
            import smplx as smplx_pkg
        except ImportError as e:
            raise ImportError(
                "The `smplx` package is required for mesh rendering. "
                "Install with `pip install smplx`."
            ) from e

        body_models_dir = self.gvhmr_dir / BODY_MODELS_REL
        smplx_dir = body_models_dir / "smplx"
        smplx_npz = smplx_dir / "SMPLX_NEUTRAL.npz"
        if not smplx_npz.exists():
            raise FileNotFoundError(
                f"SMPLX_NEUTRAL.npz not found at {smplx_npz}. "
                f"Download SMPL-X v1.1 from https://smpl-x.is.tue.mpg.de and "
                f"place SMPLX_NEUTRAL.npz at that exact path."
            )

        # smplx.create() wants the parent of `smplx/`, not the .npz directly.
        # use_pca=False so we accept hand_pose as full axis-angle (we pass
        # zeros below since GVHMR doesn't predict hands).
        self._smplx_model = smplx_pkg.create(
            model_path=str(body_models_dir),
            model_type="smplx",
            gender="neutral",
            num_betas=10,
            use_pca=False,
            flat_hand_mean=True,
        ).to(self.device).eval()

        # Cache faces as a numpy int32 array (the smplx package returns int64
        # which is wasteful when we ship to a browser).
        self._smplx_faces = np.asarray(self._smplx_model.faces, dtype=np.int32)
        return self._smplx_model

    @property
    def smplx_faces(self) -> np.ndarray:
        """SMPL-X face indices, (20908, 3) int32. Triggers ensure_smplx()."""
        self.ensure_smplx()
        return self._smplx_faces

    def smplx_forward(self, smpl_g: dict, zero_transl: bool = True):
        """Run SMPL-X forward to recover vertex positions.

        Args:
            smpl_g: dict with body_pose (F,63), betas (F,10) or (10,),
                    global_orient (F,3), transl (F,3). Values are CPU tensors
                    as returned by predict(). global_orient may have its yaw
                    pre-zeroed by the caller (see lift_segment).
            zero_transl: if True, ignore the predicted translation and place
                    every body at the origin. The viewer then translates each
                    mesh to its pitch position from PnLCalib. This is what
                    we want for tactical visualisation.

        Returns:
            verts: (F, 10475, 3) torch.float32 on CPU, in GVHMR's
                   gravity-aligned global frame (with yaw zeroed if the
                   caller pre-processed global_orient). Callers in
                   lift_segment apply a Y-floor translation before base64
                   encoding so feet sit at Y=0 for the Three.js viewer.

        Note on the explicit zero-pose tensors below:
            smplx.create(...) registers jaw_pose, leye_pose, reye_pose,
            left_hand_pose, right_hand_pose, expression as nn.Parameters
            with batch_size=1. Inside SMPLX.forward() those defaults get
            torch.cat()'d along the joints dimension together with our
            (F, 63) body_pose, producing "Expected size F but got size 1
            for tensor number 2" because dim 0 (batch) must match across the
            list. We have to pass them in explicitly with batch_size=F.
        """
        import torch
        self.ensure_smplx()
        F_len = smpl_g["body_pose"].shape[0]
        dev = self.device

        body_pose     = smpl_g["body_pose"].to(dev)               # (F, 63)
        global_orient = smpl_g["global_orient"].to(dev)           # (F, 3)
        betas         = smpl_g["betas"].to(dev)
        if betas.ndim == 1:
            betas = betas.unsqueeze(0).expand(F_len, -1).contiguous()  # (F, 10)
        if zero_transl:
            transl = torch.zeros(F_len, 3, device=dev)
        else:
            transl = smpl_g["transl"].to(dev)                     # (F, 3)

        # Zero filler for unused SMPL-X components, sized to match our batch.
        # GVHMR doesn't predict any of these so passing zeros is correct.
        z_pose3 = torch.zeros(F_len, 3,  device=dev)
        z_handL = torch.zeros(F_len, 45, device=dev)
        z_handR = torch.zeros(F_len, 45, device=dev)
        z_expr  = torch.zeros(F_len, 10, device=dev)

        with torch.no_grad():
            out = self._smplx_model(
                body_pose       = body_pose,
                global_orient   = global_orient,
                betas           = betas,
                transl          = transl,
                jaw_pose        = z_pose3,
                leye_pose       = z_pose3,
                reye_pose       = z_pose3,
                left_hand_pose  = z_handL,
                right_hand_pose = z_handR,
                expression      = z_expr,
            )
        return out.vertices.detach().cpu()  # (F, 10475, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Yaw helpers — strip the heading from global_orient for the SMPL-X forward,
# AND emit the original yaw per frame so the viewer can reconstruct the real
# body facing direction (including backpedaling / strafing).
#
# Why this exists:
#   GVHMR predicts each track in its own gravity-aligned global frame, but
#   the frame is ANCHORED to that track's first frame (the gravity head infers
#   it independently per sequence). The resulting absolute heading is
#   arbitrary across tracks — Player A's "forward" axis and Player B's
#   "forward" axis are unrelated even in the same video.
#
#   However, GVHMR's RELATIVE rotation across frames within a single track
#   IS faithful: if the player turns 90° in the real world, GVHMR's predicted
#   yaw changes by 90° in its own frame. We exploit this:
#     - Server: emit per-frame gvhmr_yaw (the Y-axis component of GVHMR's
#       global_orient) so the viewer knows how the body has rotated relative
#       to the track's first frame.
#     - Server: STILL zero the yaw before SMPL-X forward, so the rendered
#       body comes out facing a canonical direction. This decouples the
#       mesh from any per-track gravity-frame anchor.
#     - Viewer: per track, compute one calibration constant C by aligning
#       gvhmr_yaw against pitch-velocity direction at the track's fastest-
#       moving frames (where body facing ≈ velocity direction with high
#       confidence). Then for every frame:
#           mesh.rotation.y = gvhmr_yaw[i] + C
#       The body's pitch-frame heading then follows GVHMR's prediction
#       across the whole track — including backpedaling, side-stepping, and
#       sharp turns that the velocity-only heuristic gets wrong.
#
# Decomposition convention:
#   Euler "YXZ" — order applied to a vector is Z, then X, then Y. With this
#   convention the FIRST Euler component is the yaw (rotation around Y),
#   applied last. Pitch and roll are preserved (body lean during running).
# ─────────────────────────────────────────────────────────────────────────────

def _extract_yaw(global_orient):
    """Per-frame yaw (radians) from axis-angle global_orient.

    Args:
        global_orient: (F, 3) axis-angle tensor.
    Returns:
        (F,) tensor of yaw angles in radians.

    The yaw we want is the heading of the body's forward axis (+Z in SMPL-X
    canonical pose) projected onto the world XZ plane. For any rotation R:

        R @ (0, 0, 1) = (R[0,2], R[1,2], R[2,2])

    so the heading is `atan2(R[0,2], R[2,2])`. This is exactly the Y
    component of a YXZ Euler decomposition, decoupled from body pitch and
    roll — a player leaning forward at 30° while running still has a clean
    heading reading, which is essential because the viewer's per-track
    calibration constant assumes yaw varies only with true heading.

    DO NOT switch this to `atan2(R[0,2], R[0,0])`: that formula is correct
    only when pitch = roll = 0, and silently corrupts the yaw signal for
    every running/leaning player. atan2 has no singularity at any argument
    pair (including (0, 0)) so this is fully safe on ARM/Apple Silicon.
    """
    from pytorch3d.transforms import axis_angle_to_matrix
    R = axis_angle_to_matrix(global_orient)            # (F, 3, 3)
    return torch.atan2(-R[:, 0, 2], -R[:, 2, 2])         # (F,)


def _zero_yaw(global_orient):
    """Strip yaw from a per-frame global_orient tensor.

    Args:
        global_orient: (F, 3) axis-angle tensor.
    Returns:
        (F, 3) axis-angle tensor with the yaw component zeroed.

    Yaw is extracted with the same formula as `_extract_yaw` (heading of
    the +Z body axis: atan2(R[0,2], R[2,2])). A pure-Y rotation matrix is
    built from that yaw and its transpose is left-multiplied onto R, so
    R_no_yaw = R_y(-yaw) @ R keeps body pitch and roll intact while
    removing only the heading component.

    The previous formula here was atan2(R[0,2], R[0,0]); it appeared to
    work for upright/standing bodies but mixed yaw with pitch/roll for
    any leaning body, producing per-frame heading noise that the viewer's
    constant-offset calibration cannot remove.
    """
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
    R   = axis_angle_to_matrix(global_orient)          # (F, 3, 3)
    yaw = torch.atan2(R[:, 0, 2], R[:, 2, 2])          # (F,)  — heading of +Z
    cos_y = torch.cos(yaw)                             # (F,)
    sin_y = torch.sin(yaw)                             # (F,)
    zeros = torch.zeros_like(yaw)
    ones  = torch.ones_like(yaw)
    # Pure-Y rotation matrix for each frame
    R_yaw = torch.stack([
        torch.stack([ cos_y, zeros, sin_y], dim=-1),
        torch.stack([ zeros,  ones, zeros], dim=-1),
        torch.stack([-sin_y, zeros, cos_y], dim=-1),
    ], dim=-2)                                         # (F, 3, 3)
    R_no_yaw = R_yaw.transpose(-1, -2) @ R             # cancel yaw component
    return matrix_to_axis_angle(R_no_yaw)


# ─────────────────────────────────────────────────────────────────────────────
# Per-segment lift — the actual GVHMR forward for one contiguous track.
# ─────────────────────────────────────────────────────────────────────────────

def lift_segment(runner: GVHMRRunner, frame_cache: FrameCache,
                 segment: dict, K_full, verts_dtype: str = "off") -> dict:
    """Run GVHMR on one contiguous track segment.

    Args:
        runner:      GVHMRRunner (model + extractor + optional smplx)
        frame_cache: video frames loaded in memory
        segment:    {"frames":[...], "bboxes":[...], "kpts":[...], "kind", "team"}
        K_full:      torch.FloatTensor (3, 3) — image intrinsics
        verts_dtype: "off" | "fp16" | "fp32". When not "off", we:
                       - extract per-frame gvhmr_yaw from global_orient,
                       - zero the yaw in global_orient and run SMPL-X forward,
                       - floor-translate so the lowest vertex across all
                         frames sits at Y=0.
                     Each per-frame output dict gains both gvhmr_yaw and
                     verts_b64. The viewer combines gvhmr_yaw with a per-
                     track calibration constant from pitch-velocity to drive
                     mesh.rotation.y, so the body's real facing direction
                     (including backpedaling) shows up on the pitch.

    Returns:
        {
          "frames":       segment["frames"],
          "smpl_params":  list of dicts, len == segment len,
                          each {"body_pose":[...], "betas":[...],
                                "global_orient":[...], "transl":[...],
                                "gvhmr_yaw": float (if verts_dtype != "off"),
                                "verts_b64": "..."   (if verts_dtype != "off")},
        }
    """
    import torch
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    from hmr4d.utils.geo_transform import compute_cam_angvel

    F = len(segment["frames"])

    # ── 1. Bboxes → xys with the same 1.2× enlarge as the demo
    bbx_xyxy = torch.tensor(segment["bboxes"], dtype=torch.float32)  # (F, 4)
    bbx_xys  = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2)     # (F, 3)

    # ── 2. Keypoints: pipeline.py gives [x_px, y_px, conf] (or None for
    # interpolated frames where we don't have ViTPose). Convert to (F,17,3).
    kp2d = np.zeros((F, 17, 3), dtype=np.float32)
    for i, k in enumerate(segment["kpts"]):
        if k is None:
            continue  # leave as zeros — confidence 0 tells GVHMR to ignore
        arr = np.asarray(k, dtype=np.float32)  # (17, 3)
        if arr.shape != (17, 3):
            # Defensive: skip malformed entries rather than crash
            continue
        kp2d[i] = arr
    kp2d_t = torch.from_numpy(kp2d)

    # ── 3. Camera intrinsics, replicated to (F, 3, 3)
    K_seq = K_full.unsqueeze(0).expand(F, 3, 3).contiguous().clone()

    # ── 4. Static camera → identity R_w2c → zero cam_angvel
    R_w2c = torch.eye(3).repeat(F, 1, 1)
    cam_angvel = compute_cam_angvel(R_w2c)

    # ── 5. Read video frames for this segment, extract HMR2 features
    frames_np = frame_cache.slice(segment["frames"])  # (F, H, W, 3) uint8
    f_imgseq = runner.extract_features(frames_np, bbx_xys)  # (F, 1024)

    # ── 6. Run GVHMR
    data = {
        "length":     torch.tensor(F),
        "kp2d":       kp2d_t,
        "bbx_xys":    bbx_xys,
        "K_fullimg":  K_seq,
        "cam_angvel": cam_angvel,
        "f_imgseq":   f_imgseq,
    }
    pred = runner.predict(data, static_cam=True)

    smpl = pred["smpl_params_incam"]
    # Expected keys: body_pose, betas, global_orient, transl
    # Shapes:        (F,63)     (F,10) (F,3)          (F,3)
    body_pose      = smpl["body_pose"]
    betas          = smpl["betas"]
    global_orient  = smpl["global_orient"]
    transl         = smpl["transl"]

    # Some endecoders return betas as (10,) rather than (F,10). Normalise.
    if betas.ndim == 1:
        betas = betas.unsqueeze(0).expand(F, -1).contiguous()

    smpl_per_frame = []
    for i in range(F):
        smpl_per_frame.append({
            "body_pose":     [round(float(x), 5) for x in body_pose[i].tolist()],
            "global_orient": [round(float(x), 5) for x in global_orient[i].tolist()],
            "betas":         [round(float(x), 5) for x in betas[i].tolist()],
            "transl":        [round(float(x), 5) for x in transl[i].tolist()],
        })

    # ── 7. Optional verts: SMPL-X forward on the gravity-aligned global params
    # with zero translation (viewer places the mesh at pitch_pos from PnLCalib)
    # AND yaw stripped from global_orient. Per-frame gvhmr_yaw is captured
    # BEFORE the zeroing so the viewer can reconstruct the body's real facing
    # direction (relative rotation within a track is reliable, even though
    # the absolute frame anchor is per-track-arbitrary). Vertices are floor-
    # translated so the lowest point across all frames sits at Y=0 — the
    # viewer can then position the mesh at pitch_pos with Y=0 and feet will
    # rest on the grass with no additional offset needed.
    if verts_dtype != "off":
        smpl_g = dict(pred["smpl_params_global"])  # shallow copy; about to mutate

        # Capture per-frame yaw from GVHMR's full global_orient BEFORE we
        # strip it for the verts forward. These per-frame values are what
        # the viewer reads to track body rotation within the track.
        gvhmr_yaws = _extract_yaw(smpl_g["global_orient"]).detach().cpu().tolist()

        # Strip yaw so the mesh comes out facing a canonical direction
        # regardless of GVHMR's per-track gravity-frame anchoring. Pitch and
        # roll (body lean) are preserved.
        smpl_g["global_orient"] = _zero_yaw(smpl_g["global_orient"])

        verts = runner.smplx_forward(smpl_g, zero_transl=True)  # (F,10475,3)
        verts_np = verts.numpy().astype(np.float32)  # work in fp32 before floor

        # Floor-translate: shift the whole sequence up so the lowest vertex
        # across ALL frames lands at Y=0. Using the global min (not per-frame)
        # keeps body height consistent — a per-frame floor would make the player
        # visibly float/sink as pose changes between frames.
        y_min = verts_np[:, :, 1].min()
        verts_np[:, :, 1] -= y_min  # feet now sit at Y=0
        verts_np[:, :, 0] *= -1     # mirror X to match Three.js handedness

        if verts_dtype == "fp16":
            verts_np = verts_np.astype(np.float16)
        elif verts_dtype == "fp32":
            pass  # already fp32
        else:
            raise ValueError(
                f"verts_dtype must be 'off', 'fp16', or 'fp32', got {verts_dtype}"
            )
        for i in range(F):
            smpl_per_frame[i]["gvhmr_yaw"] = round(float(gvhmr_yaws[i]), 5)
            smpl_per_frame[i]["verts_b64"] = base64.b64encode(
                verts_np[i].tobytes()
            ).decode("ascii")

    return {"frames": list(segment["frames"]), "smpl_params": smpl_per_frame}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry — read result.json, run all tracks, write result_mesh.json.
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_video_path(result_path: str, override: str = None) -> str:
    """Locate the source video. If --video was passed, use that. Else look
    in ../uploads/<job_id>_*.* by job_id (the prefix of result_path's stem)."""
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"--video not found: {override}")
        return override

    rp = Path(result_path)
    job_id = rp.stem.split("_")[0]   # outputs/<job_id>.json or <job_id>_3d.json
    candidates = []
    # Sibling 'uploads' is where main.py writes uploads
    for uploads_dir in [rp.parent.parent / "uploads", Path("uploads")]:
        if uploads_dir.is_dir():
            for ext in (".mp4", ".mov", ".mkv", ".avi",
                        ".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                candidates += list(uploads_dir.glob(f"{job_id}_*{ext}"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-discover the video for job '{job_id}'. "
            f"Pass --video <path> explicitly."
        )
    return str(candidates[0])


def process(result_path: str, video_path: str = None,
            out_path: str = None, device_pref: str = None,
            max_tracks: int = MAX_TRACKS_DEFAULT,
            verts_dtype: str = VERTS_DTYPE_DEFAULT,
            on_progress=None) -> dict:
    """Top-level: lift one result.json into a mesh JSON.

    Args:
        result_path:  outputs/<job_id>.json from pipeline.py
        video_path:   source video; auto-discovered from result_path if None
        out_path:     where to write the mesh JSON; defaults to
                      <result_stem>_mesh.json next to result_path
        device_pref:  "auto" | "cuda" | "mps" | "cpu"; overrides DEVICE_PREF
        max_tracks:   if > 0, lift at most this many tracks (longest segments
                      first). 0 means lift every track that passes filters.
                      Useful for fast viewer iteration: --max-tracks 6 keeps
                      the JSON under ~25 MB.
        verts_dtype:  "off" | "fp16" | "fp32". "off" emits SMPL-X parameters
                      only (small JSON). "fp16" / "fp32" also embed base64
                      vertex positions per frame so a browser viewer can
                      render the mesh without doing SMPL-X math.

    Returns:
        the mesh result dict (also written to disk)

    Notes:
        - Resolves and adds the GVHMR repo to sys.path before any hmr4d.*
          imports, so callers don't need to do it themselves.
        - Temporarily chdirs to the GVHMR repo for the duration of GVHMR work,
          restores cwd before returning. Safe to call from main.py's worker
          thread without breaking the FastAPI server's relative paths.
        - Hydra is reset before init so repeated calls in the same process
          (e.g. multiple jobs in one FastAPI worker lifetime) succeed.
    """
    if verts_dtype not in {"off", "fp16", "fp32"}:
        raise ValueError(
            f"verts_dtype must be 'off', 'fp16', or 'fp32', got {verts_dtype!r}"
        )

    device = _pick_device(device_pref or DEVICE_PREF)
    print(f"\n[lift_gvhmr] device      = {device}")
    print(f"[lift_gvhmr] verts        = {verts_dtype}")
    print(f"[lift_gvhmr] max_tracks   = {max_tracks if max_tracks else 'all'}")

    # ── Locate result.json + video (use absolute paths so chdir below
    #    doesn't break our reads/writes)
    result_path = str(Path(result_path).resolve())
    with open(result_path, "r") as f:
        result = json.load(f)

    video_path = _resolve_video_path(result_path, video_path)
    video_path = str(Path(video_path).resolve())
    print(f"[lift_gvhmr] result : {result_path}")
    print(f"[lift_gvhmr] video  : {video_path}")

    # Still-image input is detected from the source extension. GVHMR is a
    # temporal model, so the frame is replicated into a short static clip
    # (see _build_image_segments) and the lifted clip is collapsed back to the
    # single real frame below.
    is_image = Path(video_path).suffix.lower() in IMAGE_EXTS
    if is_image:
        print(f"[lift_gvhmr] input is a still image — replicating to a "
              f"{IMAGE_REPLICATE_FRAMES}-frame static clip for GVHMR")

    # ── Resolve GVHMR + put it on sys.path BEFORE chdir, so the import
    #    machinery can find it regardless of where we land.
    gvhmr_dir = _resolve_gvhmr_dir()
    _setup_gvhmr_imports(gvhmr_dir)
    print(f"[lift_gvhmr] GVHMR  : {gvhmr_dir}")

    smplx_faces_b64 = None  # set below if verts mode is on

    # All GVHMR work happens inside this chdir. cwd is restored on exit
    # (including exceptions).
    with _chdir_to_gvhmr(gvhmr_dir):
        # Imports that touch the GVHMR repo. Done here so static analysis
        # tools that miss the sys.path insert still see a normal flow.
        from hmr4d.utils.geo.hmr_cam import estimate_K  # noqa: E402

        # ── Load video frames into memory (one read serves all tracks)
        t0 = time.time()
        frame_cache = FrameCache(video_path, is_image=is_image)
        print(f"[lift_gvhmr] loaded {frame_cache.length} frames "
              f"({frame_cache.width}x{frame_cache.height}) "
              f"in {time.time()-t0:.1f}s")

        # ── Intrinsics: GVHMR's estimate_K with image dims
        K_full = estimate_K(frame_cache.width, frame_cache.height)  # (3, 3)

        # ── Load GVHMR + HMR2 models (slow — happens once per process call)
        t0 = time.time()
        runner = GVHMRRunner(gvhmr_dir, device)
        print(f"[lift_gvhmr] models ready in {time.time()-t0:.1f}s")

        # Pre-load SMPL-X (so we fail early if the model file is missing,
        # not after spending minutes on inference). Skip when verts off.
        if verts_dtype != "off":
            t0 = time.time()
            runner.ensure_smplx()
            faces = runner.smplx_faces  # (20908, 3) int32
            smplx_faces_b64 = base64.b64encode(faces.tobytes()).decode("ascii")
            print(f"[lift_gvhmr] SMPL-X model ready in {time.time()-t0:.1f}s "
                  f"({faces.shape[0]} faces)")

        # ── Per-track segments
        tracks = collect_tracks(result)
        print(f"[lift_gvhmr] {len(tracks)} candidate tracks "
              f"(kind in {sorted(KIND_FILTER)})")

        all_segments = []
        if is_image:
            # One real observation per track; replicate into a static clip so
            # GVHMR's temporal transformer has context. MIN_TRACK_LEN is met
            # by construction, so only the bbox-height filter applies.
            all_segments = _build_image_segments(tracks, IMAGE_REPLICATE_FRAMES)
            print(f"[lift_gvhmr] {len(all_segments)} image segments built "
                  f"({IMAGE_REPLICATE_FRAMES} frames each, bbox_h>={MIN_BBOX_H_PX}px)")
        else:
            for tid, track in tracks.items():
                segs = split_into_segments(track)
                for seg in segs:
                    if segment_passes_filters(seg, MIN_TRACK_LEN, MIN_BBOX_H_PX):
                        all_segments.append((tid, seg))
            print(f"[lift_gvhmr] {len(all_segments)} segments pass filters "
                  f"(len>={MIN_TRACK_LEN}, bbox_h>={MIN_BBOX_H_PX}px)")

        # ── If max_tracks is set, sort by segment length DESC and keep the
        # longest N. Longest segments give the best meshes because GVHMR's
        # transformer has more temporal context to denoise.
        if max_tracks and max_tracks > 0:
            all_segments.sort(key=lambda x: len(x[1]["frames"]), reverse=True)
            all_segments = all_segments[:max_tracks]
            print(f"[lift_gvhmr] limited to {len(all_segments)} longest "
                  f"segments (--max-tracks {max_tracks})")

        # Emit initial progress now that we know the total segment count
        if on_progress and all_segments:
            on_progress(0, len(all_segments), f"Starting — {len(all_segments)} tracks to process")

        # ── Lift each segment, collecting smplx params by (tid, frame_idx)
        smplx_by_tid_frame: dict = {}  # {(tid, frame_idx): smplx_params}
        teams_by_tid: dict = {}        # {tid: team}

        t0 = time.time()
        for s_i, (tid, seg) in enumerate(all_segments, start=1):
            try:
                seg_desc = f"track {tid} ({seg['kind']}, {len(seg['frames'])} frames)"
                print(f"[lift_gvhmr] [{s_i}/{len(all_segments)}] {seg_desc}")
                if on_progress:
                    on_progress(s_i - 1, len(all_segments),
                                f"Processing {seg_desc}")
                out = lift_segment(runner, frame_cache, seg, K_full,
                                   verts_dtype=verts_dtype)
                if is_image:
                    # Collapse the replicated static clip back to the single
                    # real frame (frame_idx == 1), using the middle frame where
                    # the temporal denoiser is most settled (no edge effects).
                    sp = out["smpl_params"]
                    if sp:
                        smplx_by_tid_frame[(tid, 1)] = sp[len(sp) // 2]
                else:
                    for fr, params in zip(out["frames"], out["smpl_params"]):
                        smplx_by_tid_frame[(tid, fr)] = params
                teams_by_tid[tid] = seg["team"]
                if on_progress:
                    on_progress(s_i, len(all_segments),
                                f"Done {seg_desc}")
            except Exception as e:
                # One bad track shouldn't kill the whole run; log + continue.
                # Print full traceback so the error is debuggable from the
                # console output alone (most users won't be running with
                # a debugger attached).
                import traceback
                print(f"[lift_gvhmr]   FAIL track {tid}: "
                      f"{type(e).__name__}: {e}")
                print(f"[lift_gvhmr]   --- traceback ---")
                traceback.print_exc()
                print(f"[lift_gvhmr]   --- end traceback ---")
                continue
        print(f"[lift_gvhmr] lifting done in {time.time()-t0:.1f}s")

    # ── Merge smplx fields back into the original result structure
    #    (done outside the chdir block — we only need original `result`)
    mesh_result = json.loads(json.dumps(result))  # deep copy
    for f in mesh_result.get("frames", []):
        fi = f.get("frame_idx")
        for p in f.get("players", []):
            tid = p.get("id")
            params = smplx_by_tid_frame.get((tid, fi))
            if params is not None:
                p["smplx"] = params

    # Metadata addendum so the viewer knows what schema to expect
    md = mesh_result.setdefault("metadata", {})
    smplx_md = {
        "model":             "smplx",
        "num_betas":         10,
        "num_body_pose_dim": 63,
        "convention":        "axis_angle",
        "params_frame":      "camera",          # body_pose / global_orient / transl
        "verts_frame":       "canonical_yup",   # verts_b64: yaw=0, Y-up, floor=0
        "verts_yaw":         "zeroed",          # viewer drives heading
        "yaw_source":        "gvhmr_per_frame", # per-frame gvhmr_yaw available
        "K_fullimg":         K_full.tolist(),
        "device":            device,
        "tracks_meshed":     len(teams_by_tid),
        "segments_meshed":   len(all_segments),
        "verts_dtype":       verts_dtype,
        "is_image":          is_image,
    }
    if smplx_faces_b64 is not None:
        smplx_md["num_verts"] = 10475
        smplx_md["num_faces"] = 20908
        smplx_md["faces_b64"] = smplx_faces_b64
        smplx_md["faces_dtype"] = "int32"
    md["smplx"] = smplx_md

    # ── Write (use absolute path — cwd has been restored to caller's)
    if out_path is None:
        rp = Path(result_path)
        stem = rp.stem
        # Drop _3d suffix if the input was already lift_motionbert's output,
        # so we don't get foo_3d_mesh.json.
        if stem.endswith("_3d"):
            stem = stem[:-3]
        out_path = str(rp.parent / f"{stem}_mesh.json")
    out_path = str(Path(out_path).resolve())
    with open(out_path, "w") as f:
        json.dump(mesh_result, f, separators=(",", ":"))
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[lift_gvhmr] wrote {out_path} ({size_mb:.1f} MB)")
    return mesh_result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Add SMPL-X meshes to a result.json (from pipeline.py).",
    )
    parser.add_argument("result_path",
                        help="Path to result.json (e.g. outputs/af07ed52.json)")
    parser.add_argument("--video", default=None,
                        help="Source video path. Auto-discovered if omitted.")
    parser.add_argument("--out", default=None,
                        help="Output path. Defaults to <stem>_mesh.json.")
    parser.add_argument("--device", default=None,
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Override device (default: auto = cuda>mps>cpu).")
    parser.add_argument("--max-tracks", type=int, default=MAX_TRACKS_DEFAULT,
                        help="Lift at most N tracks (longest first). "
                             "0 = all tracks. Use 6 for fast viewer testing.")
    parser.add_argument("--verts", default=VERTS_DTYPE_DEFAULT,
                        choices=["off", "fp16", "fp32"],
                        help="Vertex output dtype. 'off' = params only "
                             "(small JSON). 'fp16' = ~80 MB for full clip. "
                             "'fp32' = ~150 MB, matches Kaggle format.")
    args = parser.parse_args()

    # process() handles GVHMR repo discovery, sys.path, chdir, Hydra reset.
    process(
        args.result_path,
        video_path  = args.video,
        out_path    = args.out,
        device_pref = args.device,
        max_tracks  = args.max_tracks,
        verts_dtype = args.verts,
    )


if __name__ == "__main__":
    _cli()
