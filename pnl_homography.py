"""
pnl_homography.py — PnLCalib-based homography backend.

Wraps https://github.com/mguti97/PnLCalib (clone it next to this app or set
PNLCALIB_DIR). Two HRNet-W48 models detect up to 57 pitch keypoints and 23
line classes per frame; FramebyFrameCalib matches them to a FIFA 105x68 field
template and heuristic voting over several RANSAC settings picks the best
ground-plane homography, optionally refined against the detected lines.

This wrapper exposes the same interface as homography.PitchHomography:

    get_homography(frame) -> (H, fresh, detections)
        H     3x3 mapping image pixels -> pitch meters, CORNER origin,
              x in [0, 105], y in [0, 68]   (PnLCalib world is centre-origin;
              the corner shift is applied here so the rest of the pipeline
              never sees centred coordinates)
        fresh True only if calibration succeeded and reprojection error is
              under PNL_MAX_REP_ERR_PX
    pixel_to_pitch(x, y, H) -> (px, py) meters

The pitch template is FIFA 105 x 68. pipeline.py reads PITCH_LENGTH/WIDTH
from this module so the output JSON metadata and the frontend stay in sync.

ONNX optimization
  If `models/SV_kp.onnx` and `models/SV_lines.onnx` exist next to the
  original PyTorch state-dict files, this module loads them and runs the
  forward pass through ONNX Runtime instead of plain PyTorch. Outputs are
  numerically equivalent within FP32 floating point noise (~1e-6). If
  either ONNX file is missing the original PyTorch loader is used. Export
  the ONNX files yourself; see export instructions in README_OPT.md.
"""

import os
import sys

import cv2
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
PNLCALIB_DIR   = os.environ.get("PNLCALIB_DIR", "PnLCalib")
WEIGHTS_KP     = os.path.join("models", "SV_kp")        # keypoint model weights
WEIGHTS_LINE   = os.path.join("models", "SV_lines")     # line model weights
KP_THRESHOLD   = 0.3434     # PnLCalib defaults
LINE_THRESHOLD = 0.7867
PNL_REFINE     = True       # non-linear refinement against detected lines

PNL_MAX_REP_ERR_PX = 22.0   # reject calibration above this reprojection error
                            # 12px was too strict for wide-angle/close-up shots;
                            # 18px gives a better pass rate without accepting
                            # clearly wrong calibrations. Threshold is in the
                            # SOLVER's pixel space, which is bounded above by
                            # PNL_MAX_INPUT_WIDTH (see below) — so the budget
                            # stays meaningful even for 4K stills.
PNL_MIN_KEYPOINTS  = 6      # reject frames with fewer detected landmarks —
                            # RANSAC can fabricate a plausible H from a few
                            # noise points, so demand real evidence
PNL_DEVICE         = "auto" # auto -> cuda > mps > cpu
PNL_MAX_INPUT_WIDTH = 1920  # downsample frames wider than this before the
                            # forward pass + solver. The HRNet always resizes
                            # its input to 960x540 internally, so the model
                            # sees the same image regardless; what changes is
                            # FramebyFrameCalib's working pixel space. Capping
                            # at broadcast resolution does two things:
                            #   1. The model was trained on ~1080p footage and
                            #      hallucinates keypoints less often when the
                            #      input it gets resized FROM is closer to its
                            #      training distribution (3x downsample on a
                            #      4K still produces different aliasing than
                            #      the 2x downsample on broadcast it expects).
                            #   2. PNL_MAX_REP_ERR_PX is in solver-pixel space,
                            #      so one bad keypoint at 2970-wide blows past
                            #      a 22px budget that would be fine at 1920.
                            # The returned homography is composed with the
                            # inverse scale, so callers still pass ORIGINAL
                            # frame pixels to pixel_to_pitch() — the downsample
                            # is fully internal to this module.
                            # Set to a huge value (e.g. 10000) to disable.

# Debug: print why each frame's homography was rejected (None result, no
# homography in result, or rep_err over threshold) along with the actual
# rep_err value and keypoint count. Off by default — noisy on long videos.
PNL_DEBUG_REJECTIONS = True

PITCH_LENGTH = 105.0        # FIFA template used by PnLCalib
PITCH_WIDTH  = 68.0

# centred FIFA world -> corner-origin world
_T_CORNER = np.array([[1.0, 0.0, PITCH_LENGTH / 2.0],
                      [0.0, 1.0, PITCH_WIDTH  / 2.0],
                      [0.0, 0.0, 1.0]])


def _pick_device():
    # Prefer the centralized device picker
    try:
        from runtime import pick_device
        if PNL_DEVICE != "auto":
            return PNL_DEVICE
        return pick_device()
    except ImportError:
        import torch
        if PNL_DEVICE != "auto":
            return PNL_DEVICE
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# ONNX-backed model wrapper: looks like a torch nn.Module for the call sites,
# but executes through onnxruntime. The wrapper accepts a torch tensor input
# (which is what PnLCalib's existing code produces after preprocessing) and
# returns a torch tensor output (which is what the rest of get_homography
# expects: .cpu(), slicing, etc.).
# ──────────────────────────────────────────────────────────────────────────────

class _ONNXHRNet:
    """Wraps an ONNX session so it behaves like a torch model for the
    call site `tensor_out = model(tensor_in)`.

    Performance note: we go tensor -> numpy -> ONNX -> numpy -> tensor every
    call. On CPU this is free (tensors live on host already). On CUDA/MPS
    there is one host roundtrip per call, but the HRNet forward dominates
    so the copy is negligible relative to the speedup from ORT's graph
    optimization. If this ever becomes a bottleneck, switch to ORT IOBinding.
    """

    def __init__(self, onnx_path: str, device: str):
        try:
            from runtime import make_onnx_session
            self.session = make_onnx_session(onnx_path, device)
            providers = self.session.get_providers()
        except ImportError:
            import onnxruntime as ort
            if device == "cuda":
                wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif device == "mps":
                wanted = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            else:
                wanted = ["CPUExecutionProvider"]
            available = set(ort.get_available_providers())
            providers = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.path = onnx_path
        print(f"  [pnl] ONNX session ready: {onnx_path} (providers: {providers})")

    def __call__(self, x):
        import torch
        # x is (1, 3, H, W) float tensor; ONNX models are static-shape after
        # PnLCalib's resize-to-540x960 step, so this is always the same shape.
        x_np = x.detach().cpu().numpy().astype(np.float32)
        out_np = self.session.run(None, {self.input_name: x_np})[0]
        return torch.from_numpy(out_np)

    def eval(self):
        return self

    def to(self, device):
        # Device is fixed by the ORT provider chosen at construction.
        return self


class PnLHomography:
    def __init__(self, model_path=None, device=None):
        """Signature mirrors PitchHomography(model_path, device); both args
        are ignored — paths come from the module config above."""
        repo = PNLCALIB_DIR
        if not os.path.isdir(repo):
            raise FileNotFoundError(
                f"PnLCalib repo not found at '{repo}'. Clone "
                f"https://github.com/mguti97/PnLCalib there or set the "
                f"PNLCALIB_DIR environment variable.")
        sys.path.insert(0, os.path.abspath(repo))

        import yaml
        import torch
        import torchvision.transforms as T
        from utils.utils_calib import FramebyFrameCalib

        self.torch  = torch
        self.device = _pick_device()
        print(f"  [pnl] device: {self.device}")

        cfg   = yaml.safe_load(open(os.path.join(repo, "config", "hrnetv2_w48.yaml")))
        cfg_l = yaml.safe_load(open(os.path.join(repo, "config", "hrnetv2_w48_l.yaml")))

        # ── Keypoint model: prefer ONNX if present, else PyTorch state_dict
        kp_onnx = WEIGHTS_KP + ".onnx"          # e.g. models/SV_kp.onnx
        if os.path.exists(kp_onnx):
            self.model_kp = _ONNXHRNet(kp_onnx, self.device)
        else:
            from model.cls_hrnet import get_cls_net
            print(f"  [pnl] loading keypoint model (PyTorch): {WEIGHTS_KP}")
            self.model_kp = get_cls_net(cfg)
            self.model_kp.load_state_dict(torch.load(WEIGHTS_KP, map_location="cpu"))
            self.model_kp.to(self.device).eval()
            for p in self.model_kp.parameters():
                p.requires_grad_(False)

        # ── Line model: prefer ONNX if present, else PyTorch state_dict
        line_onnx = WEIGHTS_LINE + ".onnx"      # e.g. models/SV_lines.onnx
        if os.path.exists(line_onnx):
            self.model_l = _ONNXHRNet(line_onnx, self.device)
        else:
            from model.cls_hrnet_l import get_cls_net as get_cls_net_l
            print(f"  [pnl] loading line model (PyTorch): {WEIGHTS_LINE}")
            self.model_l = get_cls_net_l(cfg_l)
            self.model_l.load_state_dict(torch.load(WEIGHTS_LINE, map_location="cpu"))
            self.model_l.to(self.device).eval()
            for p in self.model_l.parameters():
                p.requires_grad_(False)

        self.resize = T.Resize((540, 960))
        self._calib_cls = FramebyFrameCalib
        self.cam   = None          # built lazily per frame size
        self._cam_wh = None
        # Scale applied to the most recent input frame (small / orig). Stored
        # so _overlay_detections / overlay_lines can map cam-space pixels back
        # to original-frame pixels for the JSON. 1.0 means no downsample.
        self._last_scale = 1.0

        # world keypoint coordinates for the overlay, corner origin
        from utils.utils_calib import keypoint_world_coords_2D
        self.kp_world = [(x + PITCH_LENGTH / 2.0, y + PITCH_WIDTH / 2.0)
                         for x, y in keypoint_world_coords_2D]

    def reset(self):
        """Drop per-clip calibration state so a new job recalibrates from
        scratch. The HRNet weights stay loaded (expensive); only the lightweight
        FramebyFrameCalib is dropped and rebuilt lazily on the next frame. Called
        once per upload so one clip's calibration never leaks into the next."""
        self.cam = None
        self._cam_wh = None
        self._last_scale = 1.0

    # ── public ────────────────────────────────────────────────────────────────
    def get_homography(self, frame):
        from utils.utils_heatmap import (
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
            complete_keypoints, coords_to_dict)

        torch = self.torch

        # ── Pre-downsample very-high-res inputs (see PNL_MAX_INPUT_WIDTH) ──
        # We keep the public interface in ORIGINAL frame coords: the caller
        # passes the source frame in, and the returned H consumes pixels in
        # that same source-frame space. Internally though, the solver runs
        # in the downsampled space so its reprojection error budget stays
        # comparable across input resolutions.
        ih_orig, iw_orig = frame.shape[:2]
        if iw_orig > PNL_MAX_INPUT_WIDTH:
            scale = float(PNL_MAX_INPUT_WIDTH) / float(iw_orig)
            new_w = PNL_MAX_INPUT_WIDTH
            new_h = int(round(ih_orig * scale))
            frame = cv2.resize(frame, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
        self._last_scale = scale

        ih, iw = frame.shape[:2]
        if self.cam is None or self._cam_wh != (iw, ih):
            self.cam = self._calib_cls(iwidth=iw, iheight=ih, denormalize=True)
            self._cam_wh = (iw, ih)

        # preprocess exactly like PnLCalib inference.py
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        if t.shape[-1] != 960:
            t = self.resize(t)
        # Keep `t` on CPU for the ONNX path; the PyTorch path moves it to
        # device via the wrapper-agnostic .to(...) below. The _ONNXHRNet
        # wrapper handles its own .cpu() copy internally.
        if not isinstance(self.model_kp, _ONNXHRNet):
            t = t.to(self.device)
        b, c, h, w = t.shape

        with torch.inference_mode():
            hm_kp = self.model_kp(t)
            hm_l  = self.model_l(t)

        kp_coords   = get_keypoints_from_heatmap_batch_maxpool(hm_kp[:, :-1].cpu())
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(hm_l[:, :-1].cpu())
        kp_dict    = coords_to_dict(kp_coords, threshold=KP_THRESHOLD)
        lines_dict = coords_to_dict(line_coords, threshold=LINE_THRESHOLD)
        kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0],
                                                 w=w, h=h, normalize=True)

        self.cam.update(kp_dict, lines_dict)   # denormalizes in place to iw/ih

        # --- KeyError guard ---------------------------------------------------
        # PnLCalib detects some keypoints that exist in keypoints_dict but are
        # NOT in the cam.subsets['ground_plane'] subset (e.g. idx 33 is a
        # circle/arc point, not a ground-plane corner). If those indices reach
        # get_correspondences() they crash with KeyError: <idx>.
        # We remove them from the camera's internal dict before voting so the
        # solver only sees valid ground-plane points.
        gp_subset = getattr(self.cam, 'subsets', {}).get('ground_plane', {})
        if gp_subset and hasattr(self.cam, 'keypoints_dict'):
            bad_keys = [k for k in list(self.cam.keypoints_dict.keys())
                        if k not in gp_subset]
            for k in bad_keys:
                del self.cam.keypoints_dict[k]
        # ----------------------------------------------------------------------

        detections = self._overlay_detections()
        if len(detections) < PNL_MIN_KEYPOINTS:
            if PNL_DEBUG_REJECTIONS:
                print(f"  [pnl] reject: only {len(detections)} keypoints "
                      f"(need >= {PNL_MIN_KEYPOINTS})")
            return None, False, detections

        res = self.cam.heuristic_voting_ground(refine_lines=PNL_REFINE)
        if res is None or res.get("homography") is None:
            if PNL_DEBUG_REJECTIONS:
                print(f"  [pnl] reject: heuristic_voting_ground returned "
                      f"{'None' if res is None else 'no homography'} "
                      f"({len(detections)} keypoints)")
            return None, False, detections
        if res["rep_err"] is None or res["rep_err"] > PNL_MAX_REP_ERR_PX:
            if PNL_DEBUG_REJECTIONS:
                rep_err = res["rep_err"]
                rep_err_s = "None" if rep_err is None else f"{rep_err:.2f}px"
                print(f"  [pnl] reject: rep_err={rep_err_s} "
                      f"(max {PNL_MAX_REP_ERR_PX}px, {len(detections)} keypoints)")
            return None, False, detections

        if PNL_DEBUG_REJECTIONS:
            scale_note = (f", input downsampled {iw_orig}->{iw} "
                          f"(scale {scale:.3f})" if scale != 1.0 else "")
            print(f"  [pnl] accept: rep_err={res['rep_err']:.2f}px "
                  f"({len(detections)} keypoints{scale_note})")

        H_img2world_centred = res["homography"]          # image -> centred FIFA
        H = _T_CORNER @ H_img2world_centred              # image -> corner origin

        # If we pre-downsampled, H maps SMALL-frame pixels -> pitch meters.
        # Compose with the scale matrix so the returned H maps ORIGINAL-frame
        # pixels -> pitch meters, matching the interface every other caller
        # in the pipeline already expects:
        #   small_px = scale * orig_px  =>  H_orig = H_small @ diag(scale, scale, 1)
        if scale != 1.0:
            S = np.array([[scale, 0.0,   0.0],
                          [0.0,   scale, 0.0],
                          [0.0,   0.0,   1.0]])
            H = H @ S

        return H, True, detections

    def pixel_to_pitch(self, pixel_x: float, pixel_y: float, H):
        if H is None:
            return None, None
        pt = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H.astype(np.float64))
        return float(mapped[0][0][0]), float(mapped[0][0][1])

    # ── internals ─────────────────────────────────────────────────────────────
    def _overlay_detections(self):
        """Detected keypoints (already denormalized to frame pixels by
        cam.update) in the same format the old backend produced.

        cam.keypoints_dict lives in the DOWNSAMPLED pixel space we passed to
        FramebyFrameCalib. The pipeline / frontend draw these on the original
        source frame, so we map them back with 1/scale before emitting."""
        inv = 1.0 / self._last_scale
        out = []
        kd = getattr(self.cam, "keypoints_dict", None) or {}
        for key, v in kd.items():
            try:
                idx = int(key)
                if not (1 <= idx <= len(self.kp_world)):
                    continue                      # aux/derived points (>57)
                real = self.kp_world[idx - 1]
                out.append({
                    "idx":   idx,
                    "pixel": [round(float(v["x"]) * inv, 1),
                              round(float(v["y"]) * inv, 1)],
                    "real":  [round(real[0], 2), round(real[1], 2)],
                    "conf":  round(float(v.get("p", 1.0)), 3)
                })
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def overlay_lines(self):
        """Return detected line segments as a list of
        {x1,y1,x2,y2} dicts (pixel coords, already denormalized by cam.update).
        Safe to call after get_homography(); returns [] if none detected.

        As with _overlay_detections, cam.lines_dict is in downsampled-pixel
        space; we scale back to original frame pixels so overlays line up."""
        inv = 1.0 / self._last_scale
        out = []
        ld = getattr(self.cam, "lines_dict", None) or {}
        for v in ld.values():
            try:
                out.append({
                    "x1": round(float(v["x_1"]) * inv, 1),
                    "y1": round(float(v["y_1"]) * inv, 1),
                    "x2": round(float(v["x_2"]) * inv, 1),
                    "y2": round(float(v["y_2"]) * inv, 1),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return out