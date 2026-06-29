"""
video_export.py — Generate per-stage overlay videos from a result JSON + source video.

Five outputs, all written to output_dir (default: same dir as result JSON):
  <job>_detection.mp4    — BBoxes + IDs colored by role
  <job>_clustering.mp4   — BBoxes colored by team A/B/GK/ref
  <job>_keypoints.mp4    — PnLCalib keypoints + detected line segments, no players
  <job>_pnlcalib.mp4     — keypoints + lines + player boxes + 2D bird's-eye minimap
  <job>_pose.mp4         — BBoxes + ViTPose 17-pt skeleton (thin lines, small dots)

CLI:
    python video_export.py outputs/abc12345.json --video uploads/abc12345_clip.mp4
"""

import os
import json
import colorsys
from pathlib import Path

import cv2
import numpy as np

# ── COCO-17 skeleton ──────────────────────────────────────────────────────────
COCO_BONES = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# Per-joint BGR colours
COCO_KP_BGR = [
    (80,  80, 255),   # 0  nose
    (80,  80, 255),   # 1  left eye
    (80,  80, 255),   # 2  right eye
    (80, 180, 255),   # 3  left ear
    (80, 180, 255),   # 4  right ear
    (80, 255, 160),   # 5  left shoulder
    (80, 255, 160),   # 6  right shoulder
    (255, 200, 80),   # 7  left elbow
    (255, 200, 80),   # 8  right elbow
    (255, 100, 80),   # 9  left wrist
    (255, 100, 80),   # 10 right wrist
    (180,  80, 255),  # 11 left hip
    (180,  80, 255),  # 12 right hip
    (255,  80, 200),  # 13 left knee
    (255,  80, 200),  # 14 right knee
    (80,  255, 255),  # 15 left ankle
    (80,  255, 255),  # 16 right ankle
]


def _bone_color(a, b):
    ca, cb = COCO_KP_BGR[a], COCO_KP_BGR[b]
    return tuple((ca[i] + cb[i]) // 2 for i in range(3))


def _kp57_colors():
    cols = []
    for i in range(57):
        h = i / 57.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))
    return cols


KP57_BGR = _kp57_colors()

KIND_BGR = {
    "player":     (255, 100,  60),
    "goalkeeper": ( 80, 230, 130),
    "referee":    ( 40, 210, 255),
    "ball":       (255, 255, 255),
}

MM_W, MM_H = 230, 150
MM_PAD     = 8
MM_MARGIN  = 12


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bbox(frame, x1, y1, x2, y2, color, label=None, thickness=2):
    """Draw a rectangle + filled label chip above it."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        fs = 0.45
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + 2), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)


def _skeleton(frame, kpts, conf_thr=0.25, bone_thick=2, dot_r=3):
    """Draw ViTPose skeleton. kpts is [[x, y, conf], ...] with 17 entries."""
    if not kpts or len(kpts) < 17:
        return
    # Unpack safely — each entry is [x, y, conf]
    pts = []
    for k in kpts:
        try:
            pts.append((int(float(k[0])), int(float(k[1])), float(k[2])))
        except (IndexError, TypeError, ValueError):
            pts.append((0, 0, 0.0))

    for a, b in COCO_BONES:
        if a >= len(pts) or b >= len(pts):
            continue
        if pts[a][2] < conf_thr or pts[b][2] < conf_thr:
            continue
        cv2.line(frame, pts[a][:2], pts[b][:2], _bone_color(a, b), bone_thick, cv2.LINE_AA)

    for i, (x, y, c) in enumerate(pts):
        if c < conf_thr:
            continue
        col = COCO_KP_BGR[i] if i < len(COCO_KP_BGR) else (200, 200, 200)
        cv2.circle(frame, (x, y), dot_r, col,       -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), dot_r, (0, 0, 0),  1, cv2.LINE_AA)


def _pitch_keypoints(frame, kpt_list):
    """Draw PnLCalib keypoints as coloured circles with index labels."""
    for kp in (kpt_list or []):
        try:
            px, py = int(kp["pixel"][0]), int(kp["pixel"][1])
            idx    = int(kp["idx"])
        except (KeyError, TypeError, ValueError):
            continue
        col = KP57_BGR[(idx - 1) % len(KP57_BGR)]
        cv2.circle(frame, (px, py), 8, (0,   0,   0), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 6, col,            -1, cv2.LINE_AA)
        cv2.putText(frame, str(idx), (px + 9, py - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)


def _pitch_lines(frame, lines):
    """Draw pitch line segments. lines is a list of {x1,y1,x2,y2} dicts
    as stored in frame['pitch_lines'] by pipeline.py."""
    if not lines:
        return
    for v in lines:
        try:
            p1 = (int(float(v["x1"])), int(float(v["y1"])))
            p2 = (int(float(v["x2"])), int(float(v["y2"])))
        except (KeyError, TypeError, ValueError):
            continue
        cv2.line(frame, p1, p2, (0, 220, 255), 2, cv2.LINE_AA)


def _minimap(frame, f_data, team_bgr, pitch_w, pitch_h):
    """Inset a 2D bird's-eye minimap in the bottom-right corner."""
    fh, fw = frame.shape[:2]
    ox = fw - MM_W - MM_PAD * 2 - MM_MARGIN
    oy = fh - MM_H - MM_PAD * 2 - MM_MARGIN
    mw = MM_W + MM_PAD * 2
    mh = MM_H + MM_PAD * 2

    # Clamp so minimap doesn't go off-screen on small videos
    if ox < 0 or oy < 0:
        return

    # Semi-transparent dark background — use separate dst buffer (avoids
    # cv2.addWeighted in-place aliasing bug when src2 == dst)
    roi = frame[oy:oy + mh, ox:ox + mw].copy()
    dark = np.full_like(roi, (10, 18, 30))
    blended = cv2.addWeighted(dark, 0.80, roi, 0.20, 0)
    frame[oy:oy + mh, ox:ox + mw] = blended
    cv2.rectangle(frame, (ox, oy), (ox + mw, oy + mh), (51, 65, 85), 1)

    sx  = MM_W / pitch_w
    sy  = MM_H / pitch_h
    gox = ox + MM_PAD
    goy = oy + MM_PAD

    # Grass background
    cv2.rectangle(frame, (gox, goy), (gox + MM_W, goy + MM_H), (38, 112, 38), -1)

    # Pitch markings
    lc = (200, 220, 200)

    def ml(ax, ay, bx, by):
        cv2.line(frame,
                 (gox + int(ax * sx), goy + int(ay * sy)),
                 (gox + int(bx * sx), goy + int(by * sy)),
                 lc, 1)

    def mr(x, y, w, h):
        cv2.rectangle(frame,
                      (gox + int(x * sx),       goy + int(y * sy)),
                      (gox + int((x + w) * sx), goy + int((y + h) * sy)),
                      lc, 1)

    ml(0, 0, pitch_w, 0); ml(pitch_w, 0, pitch_w, pitch_h)
    ml(pitch_w, pitch_h, 0, pitch_h); ml(0, pitch_h, 0, 0)
    ml(pitch_w / 2, 0, pitch_w / 2, pitch_h)
    mr(0,            (pitch_h - 40.32) / 2, 16.5, 40.32)
    mr(pitch_w - 16.5, (pitch_h - 40.32) / 2, 16.5, 40.32)

    # Centre circle
    circle_pts = []
    for i in range(33):
        a = i / 32 * 2 * np.pi
        cx = gox + int((pitch_w / 2 + 9.15 * np.cos(a)) * sx)
        cy = goy + int((pitch_h / 2 + 9.15 * np.sin(a)) * sy)
        circle_pts.append([cx, cy])
    cv2.polylines(frame, [np.array(circle_pts, dtype=np.int32)], True, lc, 1)

    # Players
    for p in (f_data.get("players") or []):
        pp = p.get("pitch_pos")
        if not pp or len(pp) < 2:
            continue
        px = gox + int(float(pp[0]) * sx)
        py = goy + int(float(pp[1]) * sy)
        kind = p.get("kind", "player")
        if kind == "referee":
            col = (40, 210, 255)
        elif kind == "goalkeeper":
            col = (80, 230, 130)
        elif p.get("team") == 0:
            col = team_bgr[0]
        elif p.get("team") == 1:
            col = team_bgr[1]
        else:
            col = (160, 160, 160)
        cv2.circle(frame, (px, py), 4, col,       -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, (0, 0, 0),  1, cv2.LINE_AA)

    # Ball
    ball = f_data.get("ball")
    if ball and ball.get("pitch_pos") and len(ball["pitch_pos"]) >= 2:
        bx = gox + int(float(ball["pitch_pos"][0]) * sx)
        by = goy + int(float(ball["pitch_pos"][1]) * sy)
        cv2.circle(frame, (bx, by), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 3, (0,   0,   0),   1, cv2.LINE_AA)


def _status_badge(frame, text, ok):
    col = (40, 185, 90) if ok else (50, 60, 220)
    pad = 6
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(frame, (8, 8), (8 + tw + pad * 2, 8 + th + pad * 2), (0, 0, 0), -1)
    cv2.rectangle(frame, (8, 8), (8 + tw + pad * 2, 8 + th + pad * 2), col, 1)
    cv2.putText(frame, text, (8 + pad, 8 + th + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)


def _frame_counter(frame, frame_idx, ts):
    fh, fw = frame.shape[:2]
    label = f"f{frame_idx}  {ts:.2f}s"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame, (fw - tw - 12, fh - 22), (fw, fh), (0, 0, 0), -1)
    cv2.putText(frame, label, (fw - tw - 6, fh - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 100, 120), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Per-stage draw functions
# ─────────────────────────────────────────────────────────────────────────────

def draw_detection(frame, f_data):
    for p in (f_data.get("players") or []):
        try:
            x1, y1, x2, y2 = p["bbox"]
        except (KeyError, TypeError, ValueError):
            continue
        col   = KIND_BGR.get(p.get("kind", "player"), KIND_BGR["player"])
        label = f"#{p.get('id', '?')} {str(p.get('kind','?'))[:3]}"
        _bbox(frame, x1, y1, x2, y2, col, label)
    ball = f_data.get("ball")
    if ball and ball.get("bbox"):
        x1, y1, x2, y2 = ball["bbox"]
        _bbox(frame, x1, y1, x2, y2, KIND_BGR["ball"], "ball")
    n = len(f_data.get("players") or [])
    _status_badge(frame, f"players:{n}", True)


def draw_clustering(frame, f_data, team_bgr):
    for p in (f_data.get("players") or []):
        try:
            x1, y1, x2, y2 = p["bbox"]
        except (KeyError, TypeError, ValueError):
            continue
        kind = p.get("kind", "player")
        if kind == "referee":
            col, lbl = KIND_BGR["referee"],    f"#{p.get('id','?')} REF"
        elif kind == "goalkeeper":
            col, lbl = KIND_BGR["goalkeeper"], f"#{p.get('id','?')} GK"
        elif p.get("team") == 0:
            col, lbl = team_bgr[0],            f"#{p.get('id','?')} A"
        elif p.get("team") == 1:
            col, lbl = team_bgr[1],            f"#{p.get('id','?')} B"
        else:
            col, lbl = (150, 150, 150),        f"#{p.get('id','?')} ?"
        _bbox(frame, x1, y1, x2, y2, col, lbl)
    ball = f_data.get("ball")
    if ball and ball.get("bbox"):
        x1, y1, x2, y2 = ball["bbox"]
        _bbox(frame, x1, y1, x2, y2, KIND_BGR["ball"], "ball")


def draw_keypoints(frame, f_data, lines_dict=None):
    _pitch_lines(frame, lines_dict)
    _pitch_keypoints(frame, f_data.get("pitch_keypoints"))
    n  = len(f_data.get("pitch_keypoints") or [])
    ok = n >= 4
    _status_badge(frame, f"kpts:{n}  {'OK' if ok else 'LOW'}", ok)


def draw_pnlcalib(frame, f_data, team_bgr, pitch_w, pitch_h, lines_dict=None):
    _pitch_lines(frame, lines_dict)
    _pitch_keypoints(frame, f_data.get("pitch_keypoints"))
    for p in (f_data.get("players") or []):
        if not p.get("pitch_pos"):
            continue
        try:
            x1, y1, x2, y2 = p["bbox"]
        except (KeyError, TypeError, ValueError):
            continue
        kind = p.get("kind", "player")
        if kind == "referee":
            col = KIND_BGR["referee"]
        elif kind == "goalkeeper":
            col = KIND_BGR["goalkeeper"]
        elif p.get("team") == 0:
            col = team_bgr[0]
        elif p.get("team") == 1:
            col = team_bgr[1]
        else:
            col = (150, 150, 150)
        _bbox(frame, x1, y1, x2, y2, col, f"#{p.get('id','?')}")
    homo_ok = bool(f_data.get("homography_available"))
    _status_badge(frame, f"homo:{'OK' if homo_ok else 'NO'}", homo_ok)
    if homo_ok:
        _minimap(frame, f_data, team_bgr, pitch_w, pitch_h)


def draw_pose(frame, f_data):
    for p in (f_data.get("players") or []):
        if p.get("kind") == "ball":
            continue
        try:
            x1, y1, x2, y2 = p["bbox"]
        except (KeyError, TypeError, ValueError):
            continue
        col = KIND_BGR.get(p.get("kind", "player"), KIND_BGR["player"])
        _bbox(frame, x1, y1, x2, y2, col, f"#{p.get('id','?')}", thickness=1)
        if p.get("kpts"):
            _skeleton(frame, p["kpts"])


# ─────────────────────────────────────────────────────────────────────────────
# Main export engine
# ─────────────────────────────────────────────────────────────────────────────

STAGES = ["detection", "clustering", "keypoints", "pnlcalib", "pose"]


def export_all(result_json_path: str,
               video_path: str,
               output_dir: str = None,
               on_progress=None,
               team_hex: tuple = ("#CC2222", "#1144CC"),
               stages: list = None):
    """
    Generate overlay videos for all (or a subset of) pipeline stages.

    Args:
        result_json_path : path to the pipeline result JSON
        video_path       : source video file
        output_dir       : where to write .mp4 files; default = same dir as JSON
        on_progress      : callable(stage_name, frames_done, total_frames)
        team_hex         : (hex_A, hex_B) jersey colours
        stages           : subset of STAGES to render; default = all five
    """
    result_json_path = Path(result_json_path).resolve()
    video_path       = Path(video_path).resolve()
    output_dir       = Path(output_dir).resolve() if output_dir else result_json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not result_json_path.exists():
        raise FileNotFoundError(f"Result JSON not found: {result_json_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    # job_id is the stem of the result JSON, e.g. "abc12345"
    job_id = result_json_path.stem

    print(f"[export] result JSON : {result_json_path}")
    print(f"[export] source video: {video_path}")

    with open(result_json_path, "r") as f:
        data = json.load(f)

    frames  = data.get("frames", [])
    meta    = data.get("metadata", {})
    pitch_w = float(meta.get("pitch_width",  105.0))
    pitch_h = float(meta.get("pitch_height",  68.0))
    out_fps = float(meta.get("playback_fps", meta.get("fps", 25.0)))
    total   = len(frames)

    if total == 0:
        raise ValueError("Result JSON contains no frames.")

    stages = stages or STAGES
    print(f"[export] stages      : {stages}")
    print(f"[export] total frames: {total}  fps={out_fps:.1f}")

    # Convert hex team colours → BGR tuples
    def hex2bgr(h: str):
        h = h.lstrip("#")
        if len(h) != 6:
            return (128, 128, 128)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    team_bgr = (hex2bgr(team_hex[0]), hex2bgr(team_hex[1]))

    # frame_idx → frame_data dict for O(1) lookup while iterating the video
    frame_map: dict = {f["frame_idx"]: f for f in frames}
    needed_raw: set = set(frame_map.keys())

    # Open source video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture cannot open: {video_path}")

    vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if vid_w == 0 or vid_h == 0:
        cap.release()
        raise RuntimeError(f"Video reports 0x0 resolution: {video_path}")

    # Create one VideoWriter per stage
    writers: dict   = {}
    out_paths: dict = {}
    for stage in stages:
        out_path = output_dir / f"{job_id}_{stage}.mp4"
        w = cv2.VideoWriter(str(out_path), fourcc, out_fps, (vid_w, vid_h))
        if not w.isOpened():
            # Try fallback codec
            w.release()
            fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
            out_path = output_dir / f"{job_id}_{stage}.avi"
            w = cv2.VideoWriter(str(out_path), fourcc2, out_fps, (vid_w, vid_h))
        writers[stage]   = w
        out_paths[stage] = out_path
        print(f"[export] {stage:12s} → {out_path.name}")

    # Load PnLCalib for line detection (keypoints + pnlcalib stages)
    # Lines come from the result JSON (stored during pipeline run).
    # No model re-run needed.
    need_lines = False
    pnl_homo   = None

    raw_frame_num = 0
    processed     = 0
    total_raw     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[export] scanning {total_raw} raw frames…")

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            raw_frame_num += 1

            if raw_frame_num not in needed_raw:
                continue

            f_data    = frame_map[raw_frame_num]
            processed += 1

            lines_dict = None

            for stage in stages:
                out = bgr.copy()
                lines = f_data.get("pitch_lines") or []
                if   stage == "detection":
                    draw_detection(out, f_data)
                elif stage == "clustering":
                    draw_clustering(out, f_data, team_bgr)
                elif stage == "keypoints":
                    draw_keypoints(out, f_data, lines)
                elif stage == "pnlcalib":
                    draw_pnlcalib(out, f_data, team_bgr, pitch_w, pitch_h, lines)
                elif stage == "pose":
                    draw_pose(out, f_data)

                _frame_counter(out, raw_frame_num, f_data.get("timestamp", 0.0))
                writers[stage].write(out)

            # Report progress once per frame (not per stage)
            if on_progress:
                on_progress("rendering", processed, total)

            if processed % 50 == 0:
                pct = processed / total * 100
                print(f"[export] {processed}/{total} frames ({pct:.0f}%)")

    finally:
        cap.release()
        for stage, w in writers.items():
            w.release()
            path = out_paths[stage]
            if path.exists():
                size_mb = os.path.getsize(str(path)) / 1024 / 1024
                print(f"[export] ✓ {stage}: {path.name}  ({size_mb:.1f} MB)")
            else:
                print(f"[export] ✗ {stage}: output file not found")

    print(f"[export] done — {processed}/{total} frames rendered")
    return {stage: str(out_paths[stage]) for stage in stages}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Generate per-stage overlay videos")
    ap.add_argument("result_json", help="Path to result JSON from pipeline.py")
    ap.add_argument("--video",   default=None,   help="Source video (auto-discovered if omitted)")
    ap.add_argument("--out",     default=None,   help="Output directory (default: same dir as JSON)")
    ap.add_argument("--stages",  nargs="+",      default=None, choices=STAGES)
    ap.add_argument("--team-a",  default="#CC2222")
    ap.add_argument("--team-b",  default="#1144CC")
    args = ap.parse_args()

    vp = args.video
    if vp is None:
        job = Path(args.result_json).stem.split("_")[0]
        candidates = []
        for ext in (".mp4", ".mov", ".mkv", ".avi"):
            candidates += glob.glob(f"uploads/{job}_*{ext}")
        if not candidates:
            raise FileNotFoundError(
                f"Cannot auto-discover video for job '{job}'. Pass --video.")
        vp = candidates[0]

    export_all(args.result_json, vp,
               output_dir=args.out,
               team_hex=(args.team_a, args.team_b),
               stages=args.stages)
