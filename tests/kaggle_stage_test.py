"""
kaggle_stage_test.py — stage-by-stage visual test of the full pipeline on
Kaggle (GPU notebook), with an inline video preview after each stage.

HOW TO USE ON KAGGLE
  1. Create a new Kaggle Notebook, set Accelerator = GPU (P100 recommended).
  2. Upload this app/ folder (+ models/, PnLCalib/, MotionBERT/) as a Kaggle
     Dataset and attach it to the notebook, OR git clone your repo in a cell.
  3. Copy each "# %% CELL: ..." block below into its own notebook cell, in
     order, and run them top to bottom. Each stage prints progress and ends
     with an inline Video(...) preview of that stage's output.
  4. Adjust VIDEO_PATH and CLIP_SECONDS in the first cell.

WHAT EACH STAGE SHOWS
  Stage 0  - trims the input video to a short clip (CLIP_SECONDS) so the
             whole notebook runs in a few minutes instead of the full match.
  Stage 1  - runs the full pipeline (pipeline.process_video) on the trimmed
             clip: detection+tracking, homography, ResNet-26 team clustering,
             ViTPose. Produces the result JSON.
  Stage 2  - renders DETECTION + TRACKING overlay: bbox, kind, track id.
  Stage 3  - renders PITCH KEYPOINTS + HOMOGRAPHY overlay: detected keypoints
             on the frame, and a top-down pitch view with player dots.
  Stage 4  - renders TEAM CLUSTERING overlay: bbox colour-coded by team
             (using metadata.teams.kit_colors from the ResNet-26 clustering).
  Stage 5  - renders VITPOSE overlay: 17 COCO skeleton keypoints drawn on
             each player/goalkeeper.
  Stage 6  - runs MotionBERT 3D lifting (lift_motionbert.main) on the result
             JSON, then renders a simple top-down 3D-skeleton-position debug
             view (root xy + a side-view height plot) so you can sanity-check
             the lift without needing the full Three.js frontend.

All renders write to /kaggle/working/ and are displayed inline with
IPython.display.Video(embed=True) after re-encoding to H.264 (browsers in
Kaggle's notebook output often won't play raw mp4v from cv2.VideoWriter).
"""

# %% CELL: 0 — setup, config, trim input clip
import os, sys, json, subprocess
import cv2
import numpy as np
from IPython.display import Video, display

# ---- EDIT THESE ----
APP_DIR      = "/kaggle/working/app"          # where you placed/cloned the app/ folder
VIDEO_PATH   = "/kaggle/input/your-dataset/sample_2.mp4"  # input video
CLIP_SECONDS = 10                             # how much of the video to test
OUT_DIR      = "/kaggle/working/stage_outputs"
# ---------------------

os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)   # pipeline.py uses relative paths (models/, PnLCalib/, etc.)

CLIP_PATH = os.path.join(OUT_DIR, "clip.mp4")
subprocess.run([
    "ffmpeg", "-y", "-i", VIDEO_PATH,
    "-t", str(CLIP_SECONDS),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
    CLIP_PATH
], check=True)

cap = cv2.VideoCapture(CLIP_PATH)
print("Clip:", CLIP_PATH)
print("  size  :", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("  fps   :", cap.get(cv2.CAP_PROP_FPS))
print("  frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
cap.release()


def to_h264(in_path, out_path):
    """Re-encode for inline browser playback (cv2.VideoWriter's mp4v often
    won't play in the notebook output)."""
    subprocess.run([
        "ffmpeg", "-y", "-i", in_path,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        out_path
    ], check=True, capture_output=True)
    return out_path


def show_video(path, width=720):
    h264_path = path.replace(".mp4", "_h264.mp4")
    to_h264(path, h264_path)
    display(Video(h264_path, embed=True, width=width))


# %% CELL: 1 — run the full pipeline on the trimmed clip
from pipeline import process_video

RESULT_JSON = os.path.join(OUT_DIR, "result.json")

print("Running pipeline (this loads YOLO, PnLCalib, ResNet-26, ViTPose)...")
result = process_video(CLIP_PATH, on_progress=lambda p: print(f"  {p['percent']}%"))

with open(RESULT_JSON, "w") as f:
    json.dump(result, f)

print("\nMetadata:")
print(json.dumps(result["metadata"], indent=2)[:1500])
print(f"\nProcessed {len(result['frames'])} frames -> {RESULT_JSON}")


# %% CELL: 2 — Stage: detection + tracking overlay
KIND_COLOR = {
    "player":     (255, 200,   0),   # cyan-ish (BGR)
    "goalkeeper": (0,   165, 255),   # orange
    "referee":    (0,     0, 255),   # red
}

cap = cv2.VideoCapture(CLIP_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
playback_fps = result["metadata"]["playback_fps"]

out_path = os.path.join(OUT_DIR, "stage2_detection.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), playback_fps, (w, h))

frame_by_idx = {f["frame_idx"]: f for f in result["frames"]}
raw_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    raw_idx += 1
    f = frame_by_idx.get(raw_idx)
    if f is None:
        continue
    out = frame.copy()
    for p in f["players"]:
        x1, y1, x2, y2 = p["bbox"]
        color = KIND_COLOR.get(p["kind"], (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{p['kind'][:3]}#{p['id']} {p['conf']:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if f["ball"]:
        bx1, by1, bx2, by2 = f["ball"]["bbox"]
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
    writer.write(out)
writer.release()
cap.release()
print("Stage 2 done:", out_path)
show_video(out_path)


# %% CELL: 3 — Stage: pitch keypoints + homography (top-down view side by side)
PITCH_L = result["metadata"]["pitch_width"]    # PnLHomography names: length=x, width=y
PITCH_W = result["metadata"]["pitch_height"]
TOPDOWN_W, TOPDOWN_H = 600, int(600 * PITCH_W / PITCH_L)

cap = cv2.VideoCapture(CLIP_PATH)
out_path = os.path.join(OUT_DIR, "stage3_homography.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), playback_fps,
                          (w + TOPDOWN_W, max(h, TOPDOWN_H)))

def pitch_to_topdown(px, py):
    return int(px / PITCH_L * TOPDOWN_W), int(py / PITCH_W * TOPDOWN_H)

raw_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    raw_idx += 1
    f = frame_by_idx.get(raw_idx)
    if f is None:
        continue

    left = frame.copy()
    # draw detected pitch keypoints
    for kp in f.get("pitch_keypoints", []) or []:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(left, (x, y), 4, (0, 255, 0), -1)

    canvas = np.full((max(h, TOPDOWN_H), w + TOPDOWN_W, 3), 40, dtype=np.uint8)
    canvas[:h, :w] = left

    # top-down pitch rectangle
    ox, oy = w, 0
    cv2.rectangle(canvas, (ox, oy), (ox + TOPDOWN_W, oy + TOPDOWN_H), (255, 255, 255), 1)

    if f["homography_available"]:
        cv2.putText(canvas, "homography: OK", (ox + 10, oy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for p in f["players"]:
            if not p["pitch_pos"]:
                continue
            tx, ty = pitch_to_topdown(p["pitch_pos"][0], p["pitch_pos"][1])
            color = KIND_COLOR.get(p["kind"], (255, 255, 255))
            cv2.circle(canvas, (ox + tx, oy + ty), 5, color, -1)
        if f["ball"]:
            bx, by = pitch_to_topdown(f["ball"]["pitch_pos"][0], f["ball"]["pitch_pos"][1])
            cv2.circle(canvas, (ox + bx, oy + by), 4, (255, 255, 255), -1)
    else:
        cv2.putText(canvas, "homography: NONE", (ox + 10, oy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    writer.write(canvas)
cap.release()
writer.release()
print("Stage 3 done:", out_path)
show_video(out_path)


# %% CELL: 4 — Stage: team clustering overlay (ResNet-26)
kit_colors_hex = (result["metadata"].get("teams") or {}).get("kit_colors")
print("Detected kit colours:", kit_colors_hex)

def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

TEAM_COLOR = {}
if kit_colors_hex:
    for i, hx in enumerate(kit_colors_hex):
        TEAM_COLOR[i] = hex_to_bgr(hx)
TEAM_COLOR[-1] = (128, 128, 128)   # unknown
TEAM_COLOR[None] = (255, 255, 255)  # undecided / referee

cap = cv2.VideoCapture(CLIP_PATH)
out_path = os.path.join(OUT_DIR, "stage4_teams.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), playback_fps, (w, h))

raw_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    raw_idx += 1
    f = frame_by_idx.get(raw_idx)
    if f is None:
        continue
    out = frame.copy()
    for p in f["players"]:
        x1, y1, x2, y2 = p["bbox"]
        if p["kind"] == "referee":
            color = (0, 0, 255)
            label = f"ref#{p['id']}"
        else:
            team = p.get("team")
            color = TEAM_COLOR.get(team, (255, 255, 255))
            label = f"{p['kind'][:3]}#{p['id']} team={team}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    writer.write(out)
cap.release()
writer.release()
print("Stage 4 done:", out_path)
show_video(out_path)


# %% CELL: 5 — Stage: ViTPose overlay
COCO_EDGES = [
    (5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4),(0,5),(0,6)
]

cap = cv2.VideoCapture(CLIP_PATH)
out_path = os.path.join(OUT_DIR, "stage5_pose.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), playback_fps, (w, h))

raw_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    raw_idx += 1
    f = frame_by_idx.get(raw_idx)
    if f is None:
        continue
    out = frame.copy()
    for p in f["players"]:
        x1, y1, x2, y2 = p["bbox"]
        color = KIND_COLOR.get(p["kind"], (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        kpts = p.get("kpts")
        if not kpts:
            continue
        for (a, b) in COCO_EDGES:
            xa, ya, sa = kpts[a]
            xb, yb, sb = kpts[b]
            if sa > 0.3 and sb > 0.3:
                cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 255), 2)
        for (x, y, s) in kpts:
            if s > 0.3:
                cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)
    writer.write(out)
cap.release()
writer.release()
print("Stage 5 done:", out_path)
show_video(out_path)


# %% CELL: 6 — Stage: MotionBERT 3D lift + debug views
from lift_motionbert import main as lift_main

lift_main(RESULT_JSON)
RESULT_3D_JSON = RESULT_JSON.replace(".json", "_3d.json")

with open(RESULT_3D_JSON) as f:
    result3d = json.load(f)

frame_by_idx_3d = {fr["frame_idx"]: fr for fr in result3d["frames"]}

# Debug view: top-down (root X vs root Z->depth proxy) + side view (root Y = height)
TD_W, TD_H = 500, 500
SIDE_W, SIDE_H = 500, 250

cap = cv2.VideoCapture(CLIP_PATH)
out_path = os.path.join(OUT_DIR, "stage6_3d.mp4")
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), playback_fps,
                          (w + TD_W, max(h, TD_H + SIDE_H)))

# H36M-17: joint 0 = Hip (root)
SCALE = 80  # px per metre-ish unit for the debug plots

raw_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    raw_idx += 1
    f = frame_by_idx_3d.get(raw_idx)
    if f is None:
        continue

    canvas = np.full((max(h, TD_H + SIDE_H), w + TD_W, 3), 30, dtype=np.uint8)
    canvas[:h, :w] = frame

    ox = w
    cv2.rectangle(canvas, (ox, 0), (ox + TD_W, TD_H), (60, 60, 60), 1)
    cv2.putText(canvas, "top-down (root X,Z)", (ox + 10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.rectangle(canvas, (ox, TD_H), (ox + SIDE_W, TD_H + SIDE_H), (60, 60, 60), 1)
    cv2.putText(canvas, "side (root Y = height)", (ox + 10, TD_H + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    for p in f["players"]:
        kpts3d = p.get("kpts3d")
        if not kpts3d:
            continue
        color = KIND_COLOR.get(p["kind"], (255, 255, 255))
        rx, ry, rz = kpts3d[0]   # root (Hip)
        tx = ox + TD_W // 2 + int(rx * SCALE)
        ty = int(TD_H // 2 + rz * SCALE)
        cv2.circle(canvas, (tx, ty), 5, color, -1)
        cv2.putText(canvas, f"#{p['id']}", (tx + 6, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        sx = ox + 20 + (p["id"] % 20) * 20
        sy = TD_H + SIDE_H - 20 - int(ry * SCALE)
        cv2.circle(canvas, (sx, max(TD_H, min(TD_H + SIDE_H, sy))), 4, color, -1)

    writer.write(canvas)
cap.release()
writer.release()
print("Stage 6 done:", out_path)
show_video(out_path)
