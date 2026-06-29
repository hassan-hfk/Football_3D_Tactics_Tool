"""
homography.py — maps player pixel positions to real pitch coordinates.

IMPORTANT: the pitch model in this project is the 29-keypoint SoccerNet model
(from the Adit-jain/Soccer_Analysis project). Its keypoint indices map to
Roboflow's SoccerPitchConfiguration template: 120m x 70m, penalty box 20.15m
deep, goal box 5.5m deep, penalty spot 11m out. Pitch origin (0,0) = top-left
corner, (120,70) = bottom-right. All values meters.

HOW THE SHAKE FIX WORKS (v2)
The old version low-passed the homography MATRIX across frames. Blending two
homographies element-wise does not produce a valid intermediate homography,
and any smoothing applied in image space lags behind camera pans — that lag is
exactly the "players shake when the camera moves" symptom.

New approach, standard in sports field registration:
  1. Track the CAMERA motion itself: sparse optical flow (Lucas-Kanade) on
     pitch-surface features between consecutive processed frames gives the
     inter-frame homography A (prev pixels -> current pixels).
  2. Carry each pitch keypoint's smoothed state through A. The states now move
     WITH the camera, so panning produces zero error in the filter.
  3. Each new model detection only corrects the small residual detector noise:
     state = flow_predicted + KPT_CORRECT_ALPHA * conf * (detected - predicted)
  4. Keypoints the model misses this frame survive on flow alone for a few
     frames (FLOW_KEEPALIVE), so the homography uses MORE anchors, not fewer.
  5. Recompute H fresh every frame from the smoothed keypoints. The matrix is
     never blended.
Camera cuts reset everything (flow fails or measured displacement is huge,
both are detected), so nothing ever glides across a cut.
"""

import cv2
import numpy as np


# Keypoint index -> real pitch coordinates (meters, 120 x 70 template)
KEYPOINT_TO_REAL_WORLD = {
     0: (  0.00,   0.00),   # sideline_top_left  (top-left corner)
     1: (  0.00,  14.50),   # big_rect_left_top_pt1   (penalty box, goal line)
     2: ( 20.15,  14.50),   # big_rect_left_top_pt2   (penalty box, far line)
     3: (  0.00,  55.50),   # big_rect_left_bottom_pt1
     4: ( 20.15,  55.50),   # big_rect_left_bottom_pt2
     5: (  0.00,  25.84),   # small_rect_left_top_pt1 (goal box, goal line)
     6: (  5.50,  25.84),   # small_rect_left_top_pt2 (goal box, far line)
     7: (  0.00,  44.16),   # small_rect_left_bottom_pt1
     8: (  5.50,  44.16),   # small_rect_left_bottom_pt2
     9: (  0.00,  70.00),   # sideline_bottom_left  (bottom-left corner)
    10: ( 29.32,  35.00),   # left_semicircle_right (penalty arc tip)
    11: ( 60.00,   0.00),   # center_line_top
    12: ( 60.00,  70.00),   # center_line_bottom
    13: ( 60.00,  25.85),   # center_circle_top
    14: ( 60.00,  44.15),   # center_circle_bottom
    15: ( 60.00,  35.00),   # field_center (centre spot)
    16: (120.00,   0.00),   # sideline_top_right (top-right corner)
    17: (120.00,  14.50),   # big_rect_right_top_pt1
    18: ( 99.85,  14.50),   # big_rect_right_top_pt2
    19: (120.00,  55.50),   # big_rect_right_bottom_pt1
    20: ( 99.85,  55.50),   # big_rect_right_bottom_pt2
    21: (120.00,  25.84),   # small_rect_right_top_pt1
    22: (114.50,  25.84),   # small_rect_right_top_pt2
    23: (120.00,  44.16),   # small_rect_right_bottom_pt1
    24: (114.50,  44.16),   # small_rect_right_bottom_pt2
    25: (120.00,  70.00),   # sideline_bottom_right (bottom-right corner)
    26: ( 90.69,  35.00),   # right_semicircle_left (penalty arc tip)
    27: ( 50.85,  35.00),   # center_circle_left
    28: ( 69.15,  35.00),   # center_circle_right
}

PITCH_LENGTH = 120.0
PITCH_WIDTH  = 70.0


# ── Tuning ────────────────────────────────────────────────────────────────────
MIN_KEYPOINTS    = 4      # maths minimum
MAX_REPROJ_ERR_M = 3.0    # was 5.0 — 5m is inside-vs-outside the penalty box
KPT_CONF         = 0.5    # keypoint confidence threshold

MIN_SPREAD_W_FRAC = 0.15  # reject keypoint sets clustered in a small region
MIN_SPREAD_H_FRAC = 0.10

# Keypoint state filtering (all in pixels)
KPT_CORRECT_ALPHA = 0.35  # how strongly a new detection corrects the
                          # flow-predicted state (scaled by detection conf).
                          # Higher = more responsive to detector, noisier.
FLOW_KEEPALIVE    = 5     # frames a keypoint survives on flow alone
KPT_OUTLIER_PX    = 40.0  # a detection this far from its flow-predicted state
                          # is ignored as a misdetection (unless many agree,
                          # which is handled by the cut check below)

# Camera cut detection
CUT_MED_DISP_FRAC = 0.06  # cut if median keypoint displacement between the
                          # flow prediction and the new detections exceeds
                          # this fraction of frame width
RESET_AFTER_GAP   = 5     # forget all state after this many bad frames

# Optical flow (camera motion between processed frames)
FLOW_MAX_FEATS = 300
FLOW_QUALITY   = 0.01
FLOW_MIN_DIST  = 12
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FLOW_MIN_INLIERS = 25     # need this many flow inliers to trust camera motion


class PitchHomography:
    def __init__(self, model_path: str, device: str = "0"):
        from ultralytics import YOLO
        print(f"  Loading pitch keypoint model: {model_path}")
        self.model          = YOLO(model_path)
        self.device         = device
        self.conf_threshold = KPT_CONF
        self._reset_state()

    def _reset_state(self):
        self.kpt_state = {}    # idx -> {"pt": np.array([x,y]), "age": int}
        self.prev_gray = None
        self.prev_pts  = None  # features for LK
        self.gap       = 0

    # ── public ────────────────────────────────────────────────────────────────
    def get_homography(self, frame):
        """
        Returns:
            H          - 3x3 frame->pitch matrix, or None if not reliable
            fresh      - True only if H passed all quality checks this frame
            detections - raw detected keypoints (pixels) for the video overlay
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. camera motion since the previous processed frame
        A = self._camera_motion(gray)            # 3x3 or None

        # 2. detect pitch keypoints with the model
        det, detections = self._detect_keypoints(frame)

        # 3. cut check: flow lost AND detections moved far -> hard reset
        if self._looks_like_cut(A, det, frame.shape[1]):
            self._reset_state()
            self.prev_gray = gray
            self._seed_flow_features(gray, frame)
            # re-seed states straight from this frame's detections
            for idx, (pt, conf) in det.items():
                self.kpt_state[idx] = {"pt": pt.copy(), "age": 0}
            return self._compute_H(frame, detections)

        # 4. propagate keypoint states through the camera motion
        self._propagate_states(A)

        # 5. correct states with new detections
        for idx, (pt, conf) in det.items():
            st = self.kpt_state.get(idx)
            if st is None:
                self.kpt_state[idx] = {"pt": pt.copy(), "age": 0}
                continue
            if np.linalg.norm(pt - st["pt"]) > KPT_OUTLIER_PX:
                continue                          # lone misdetection, ignore
            a = KPT_CORRECT_ALPHA * float(conf)
            st["pt"] = st["pt"] + a * (pt - st["pt"])
            st["age"] = 0

        # 6. expire stale flow-only keypoints
        for idx in [i for i, s in self.kpt_state.items()
                    if s["age"] > FLOW_KEEPALIVE]:
            del self.kpt_state[idx]

        # 7. prepare flow features for the next frame
        self.prev_gray = gray
        self._seed_flow_features(gray, frame)

        return self._compute_H(frame, detections)

    def pixel_to_pitch(self, pixel_x: float, pixel_y: float, H):
        if H is None:
            return None, None
        pt     = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)
        return float(mapped[0][0][0]), float(mapped[0][0][1])

    # ── internals ─────────────────────────────────────────────────────────────
    def _detect_keypoints(self, frame):
        preds = self.model.predict(frame, conf=self.conf_threshold,
                                   device=self.device, verbose=False)[0]
        det, detections = {}, []
        if preds.keypoints is None or len(preds.keypoints) == 0:
            return det, detections
        kpts_xy   = preds.keypoints.xy.cpu().numpy()[0]
        kpts_conf = preds.keypoints.conf.cpu().numpy()[0] \
                    if preds.keypoints.conf is not None else None
        for idx, (x, y) in enumerate(kpts_xy):
            conf = float(kpts_conf[idx]) if kpts_conf is not None else 1.0
            if conf >= self.conf_threshold and x > 0 and y > 0 \
                    and idx in KEYPOINT_TO_REAL_WORLD:
                det[idx] = (np.array([float(x), float(y)]), conf)
                detections.append({
                    "idx":   idx,
                    "pixel": [round(float(x), 1), round(float(y), 1)],
                    "real":  KEYPOINT_TO_REAL_WORLD[idx],
                    "conf":  round(conf, 3)
                })
        return det, detections

    def _camera_motion(self, gray):
        """Inter-frame homography from sparse LK flow on the pitch surface."""
        if self.prev_gray is None or self.prev_pts is None \
                or len(self.prev_pts) < FLOW_MIN_INLIERS:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **LK_PARAMS)
        if nxt is None:
            return None
        ok = status.ravel() == 1
        if int(ok.sum()) < FLOW_MIN_INLIERS:
            return None
        p0 = self.prev_pts[ok].reshape(-1, 2)
        p1 = nxt[ok].reshape(-1, 2)
        A, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        if A is None or mask is None or int(mask.sum()) < FLOW_MIN_INLIERS:
            return None
        return A

    def _seed_flow_features(self, gray, frame):
        """Shi-Tomasi corners on the (green) pitch surface for the next LK run.
        Mask to grass so we track the rigid pitch, not moving players' bodies —
        player pixels inside the mask are a minority and RANSAC rejects them."""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
        mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
        pts  = cv2.goodFeaturesToTrack(gray, FLOW_MAX_FEATS, FLOW_QUALITY,
                                       FLOW_MIN_DIST, mask=mask)
        self.prev_pts = pts

    def _propagate_states(self, A):
        if not self.kpt_state:
            return
        if A is None:
            # no camera estimate this frame: hold positions, but age fast so a
            # real pan without flow can't drag stale anchors around for long
            for st in self.kpt_state.values():
                st["age"] += 2
            return
        idxs = list(self.kpt_state.keys())
        pts  = np.array([self.kpt_state[i]["pt"] for i in idxs],
                        dtype=np.float32).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(pts, A).reshape(-1, 2)
        for i, idx in enumerate(idxs):
            self.kpt_state[idx]["pt"]  = moved[i]
            self.kpt_state[idx]["age"] += 1

    def _looks_like_cut(self, A, det, frame_w):
        """A camera cut shows up as: optical flow lost, AND the detected
        keypoints far from where the states expect them."""
        if not self.kpt_state or not det:
            return False
        common = [i for i in det if i in self.kpt_state]
        if len(common) < 3:
            return A is None and len(det) >= MIN_KEYPOINTS
        disp = np.median([np.linalg.norm(det[i][0] - self.kpt_state[i]["pt"])
                          for i in common])
        if A is None:
            return disp > CUT_MED_DISP_FRAC * frame_w
        return disp > 3 * CUT_MED_DISP_FRAC * frame_w   # flow OK: be tolerant

    def _compute_H(self, frame, detections):
        """Build H from the current smoothed keypoint states + quality guards."""
        usable = {i: s for i, s in self.kpt_state.items()
                  if i in KEYPOINT_TO_REAL_WORLD}
        if len(usable) < MIN_KEYPOINTS:
            return self._miss(detections)

        src = np.array([s["pt"] for s in usable.values()], dtype=np.float32)
        dst = np.array([KEYPOINT_TO_REAL_WORLD[i] for i in usable.keys()],
                       dtype=np.float32)

        h, w = frame.shape[:2]
        if (src[:, 0].max() - src[:, 0].min()) < MIN_SPREAD_W_FRAC * w or \
           (src[:, 1].max() - src[:, 1].min()) < MIN_SPREAD_H_FRAC * h:
            return self._miss(detections)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or not self._is_reliable(src, dst, H, mask):
            return self._miss(detections)

        self.gap = 0
        return H, True, detections

    def _miss(self, detections):
        self.gap += 1
        if self.gap > RESET_AFTER_GAP:
            self._reset_state()
        return None, False, detections

    def _is_reliable(self, src, dst, H, mask) -> bool:
        inliers = mask.ravel().astype(bool) if mask is not None \
                  else np.ones(len(src), dtype=bool)
        if int(inliers.sum()) < MIN_KEYPOINTS:
            return False
        s_in   = src[inliers].reshape(-1, 1, 2)
        proj   = cv2.perspectiveTransform(s_in, H).reshape(-1, 2)
        errors = np.linalg.norm(proj - dst[inliers], axis=1)
        return float(errors.mean()) <= MAX_REPROJ_ERR_M
