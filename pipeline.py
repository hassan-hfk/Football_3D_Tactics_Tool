"""
pipeline.py — player detection (ByteTrack) + pitch keypoints + homography
              + team assignment + Kalman position filtering.

Key rules:
  - Detection + tracking ALWAYS runs, every processed frame, regardless of
    homography. The detector performs well standalone and players/ball/
    referee should be reported whenever they're visible, even during camera
    angles PnLCalib can't calibrate (replays, close-ups, alternate angles).
  - Players are only PLACED ON THE PITCH (pitch_pos) when the pitch is
    actually visible in the frame (fresh homography). No stale matrix is
    ever used. Without fresh homography, pitch_pos is null and the track is
    not fed to the Kalman filter or team-side accumulation, but it still
    appears in `players` with bbox/kind/id/kpts.
  - Pitch positions go through a per-track constant-velocity Kalman filter:
    smooth movement, no lag, coasting through 1-2 missed detections, and
    rejection of physically impossible jumps from homography glitches.
  - Role (player / goalkeeper / referee / ball) is decided entirely by the
    detector — it performs well on its own and is trusted as-is, `kind`/`cls`
    are never changed after detection.
  - Team (0 / 1) is decided from a ResNet-26 embedding of the torso crop
    (k-means bootstrap, sliding-window verdict per track). Only player /
    goalkeeper tracks are considered; referees never get a team and are not
    part of the bootstrap or voting.
  - Bbox-overlap COLLISIONS are flagged each frame (any pair of player /
    goalkeeper bboxes with IoU > TEAM_COLLISION_IOU). For each flagged pair
    we call team.mark_collision(), which clears both tracks' recent vote
    windows and forces every-frame re-embedding for the next ~20 frames.
    This makes team assignment self-correct after BoT-SORT track-ID swaps
    that can happen when two players occlude each other, without flickering
    on benign close passes.

Output JSON additions vs the old version:
  player["team"] : 0 / 1 / -1(unknown kit) / null(undecided short track)
  player["snap"] : true only on frames where the position jumped for real
                   (new track / tracker re-lock). The frontend should hard-set
                   the figure there instead of gliding to it.
  metadata["teams"]: the two detected kit colours as hex RGB, so the frontend
                   can default jersey colours to the real kits.
  frame["pitch_lines"]: list of {x1,y1,x2,y2} line segment pixel coords from
                   PnLCalib, stored here so video_export.py can draw them
                   without re-running the model.

Batching (PR2)
  Per-frame, all qualifying detections are collected first, then the
  ResNet-26 embedder and ViTPose are each invoked once with the full batch
  rather than once per player. Outputs are bit-exact equivalent to the
  per-player code path because the model forward passes are independent
  across the batch dimension.

ONNX YOLO (PR2)
  If `models/player_detection_v26s.onnx` exists, Ultralytics' YOLO class
  loads the ONNX version (which it supports natively). Otherwise the
  original .pt is used. Tracking via BoT-SORT and the `track()` API is
  unchanged in either case.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from team_assign import TeamAssigner
from pitch_filter import PitchKalman
import os

# Try to import the centralized runtime config (PR1). If absent the module
# still works; it just won't print the banner or use centralized thread
# tuning. This keeps pipeline.py runnable in environments where runtime.py
# was not deployed yet.
try:
    from runtime import pick_device as _runtime_pick_device, print_banner
    _RUNTIME_OK = True
except ImportError:
    _RUNTIME_OK = False

# ── Config ────────────────────────────────────────────────────────────────────
HOMOGRAPHY_BACKEND = "pnlcalib"

PLAYER_MODEL_PATH = os.path.join("models", "player_detection_v26s.pt")
PLAYER_MODEL_ONNX = os.path.join("models", "player_detection_v26s.pt")
PITCH_MODEL_PATH  = os.path.join("models", "pitch_keypoints.pt")

DEVICE_PREF       = "auto"


def _pick_device():
    if _RUNTIME_OK:
        return _runtime_pick_device(DEVICE_PREF)
    if DEVICE_PREF != "auto":
        return DEVICE_PREF
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()

if HOMOGRAPHY_BACKEND == "pnlcalib":
    from pnl_homography import PnLHomography as _HomographyBackend
    from pnl_homography import PITCH_LENGTH, PITCH_WIDTH
else:
    from homography import PitchHomography as _HomographyBackend
    from homography import PITCH_LENGTH, PITCH_WIDTH

DET_CONF          = 0.15
PERSON_MIN_CONF   = 0.35
BALL_MIN_CONF     = 0.15
PLAYER_IMGSZ      = 1280
PROCESS_EVERY_N   = 2

TRACKER_CFG       = "botsort_reid.yaml"
GK_ZONE_FRAC      = 0.25

# Bbox-IoU threshold for flagging a pair of tracked players/goalkeepers as
# potentially colliding (mutually occluding). When two players' bboxes
# overlap by more than this fraction, BoT-SORT is prone to swapping their
# track IDs as one re-emerges from the occlusion. We pass both tracks to
# team.mark_collision() so the team assigner clears their recent vote
# windows and forces every-frame embedding for the next ~20 frames — the
# new evidence then disambiguates the post-swap identities.
#
# 0.4 catches real shoulder-to-shoulder contact and tight cover defense
# without firing on the loose mutual proximity of two teammates trotting
# parallel to each other (which typically gives IoU < 0.2 with normal
# detector bbox tightness).
TEAM_COLLISION_IOU = 0.4

CLS_BALL       = 0
CLS_GOALKEEPER = 1
CLS_PLAYER     = 2
CLS_REFEREE    = 3
KIND = {CLS_BALL: "ball", CLS_GOALKEEPER: "goalkeeper",
        CLS_PLAYER: "player", CLS_REFEREE: "referee"}

PERSON_CLASSES = [CLS_GOALKEEPER, CLS_PLAYER, CLS_REFEREE]
TRACK_BALL     = True

ENABLE_POSE     = True
POSE_ONNX       = os.path.join("models", "vitpose-b-coco.onnx")
POSE_CLASSES    = {CLS_GOALKEEPER, CLS_PLAYER, CLS_REFEREE}  # referees too,
                                                             # so their meshes
                                                             # are well-posed
POSE_EVERY_N    = 1
POSE_MIN_BBOX_H = 35

_player_model = None
_pitch_homo   = None
_pose_model   = None


def get_models():
    global _player_model, _pitch_homo, _pose_model
    if _player_model is None:
        # Prefer ONNX export if available — Ultralytics' YOLO() accepts an
        # ONNX path directly and the .track() interface is identical.
        if os.path.exists(PLAYER_MODEL_ONNX):
            print(f"  Loading player model (ONNX): {PLAYER_MODEL_ONNX}")
            _player_model = YOLO(PLAYER_MODEL_ONNX)
        else:
            print(f"  Loading player model (PyTorch): {PLAYER_MODEL_PATH}")
            _player_model = YOLO(PLAYER_MODEL_PATH)
        print(f"  Detector classes     : {_player_model.names}")
    if _pitch_homo is None:
        _pitch_homo = _HomographyBackend(PITCH_MODEL_PATH, DEVICE)
    if ENABLE_POSE and _pose_model is None:
        if os.path.exists(POSE_ONNX):
            from pose import ViTPose
            _pose_model = ViTPose(POSE_ONNX, DEVICE)
        else:
            print(f"  [pose] ONNX not found at {POSE_ONNX} — pose disabled")
    return _player_model, _pitch_homo, _pose_model


def _bbox_iou(a, b):
    """Intersection-over-union of two [x1, y1, x2, y2] bboxes. Returns 0 for
    non-overlapping pairs. Pure-Python, microseconds per call — fine for the
    O(n^2) pairwise check we run per frame on a typical 22-player scene."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union  = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _flag_team_collisions(team_observations, team, proc_idx):
    """Find pairs of tracked player/goalkeeper bboxes whose IoU exceeds
    TEAM_COLLISION_IOU and notify TeamAssigner of each pair. Called once per
    frame, before team.observe_batch(...).

    Why this matters
      BoT-SORT is robust to brief occlusions in most cases, but when two
      players' bboxes overlap heavily (shoulder-to-shoulder, sliding tackle,
      crowded set-piece) the tracker can swap their track IDs as the two
      players separate. The old team assigner permanently locked a track's
      team after 25 votes, so a post-swap track would keep showing the
      wrong team color for the rest of the video. The new sliding-window
      assigner CAN flip teams when evidence shifts, but only after enough
      new evidence accumulates. mark_collision() short-circuits that wait:
      it clears both involved tracks' vote windows so the post-collision
      embeddings dominate the next decision, and force-embeds every frame
      (no subsampling) for the next ~20 frames.

      We do NOT clear self.team[tid] here — the visible team color stays as
      it was until the new evidence accumulates and the window-majority +
      hysteresis logic flips it. That prevents flicker on every benign
      crowded frame.
    """
    n = len(team_observations)
    if n < 2:
        return
    # Pre-extract the bboxes we need to compare; reading the tuple repeatedly
    # in the inner loop is the kind of thing CPython does fine but isn't free
    # when this runs 30 times per second on every frame.
    bboxes = [obs[2] for obs in team_observations]
    tids   = [obs[0] for obs in team_observations]
    for i in range(n):
        if tids[i] < 0:
            continue
        for j in range(i + 1, n):
            if tids[j] < 0:
                continue
            if _bbox_iou(bboxes[i], bboxes[j]) > TEAM_COLLISION_IOU:
                team.mark_collision(tids[i], tids[j], proc_idx)


def _reset_tracker(model):
    """Fully reset the detector's BoT-SORT tracker so every job starts clean.

    The detector is a module-level singleton shared across jobs (get_models),
    and Ultralytics stores tracker state — active / lost tracks, ReID memory,
    Kalman motion AND the global track-ID counter — on model.predictor, keeping
    it alive between .track() calls. We rely on that WITHIN a clip (persist=True
    gives continuous tracking frame to frame), but between jobs it must be wiped
    or the first frame of a new clip is matched against the *previous* clip's
    tracks. That stale association is the ID fluctuation / garbage tracking that
    showed up on every video after the first, and the reason a server restart
    'fixed' it (a restart just rebuilds the singleton). Resetting here makes the
    upload itself do the cleanup — no restart needed.

    Note: resetting only the ID counter is NOT enough — the tracker objects keep
    their old track lists and ReID features, so we clear those too. On the first
    job the predictor doesn't exist yet, so there's simply nothing to clear.
    """
    # 1) Global ID counter -> tracks number from #1 again each job.
    try:
        from ultralytics.trackers.basetrack import BaseTrack
        BaseTrack.reset_id()
    except Exception as e:
        print(f"  [tracker] id-counter reset skipped: {e}")

    # 2) Wipe each live tracker's internal state. BoT-SORT keeps THREE kinds
    #    of cross-frame state and all must go or the new clip inherits the old:
    #      - track lists + ReID features  (reset() / manual clear)
    #      - the global track-ID counter  (reset above + reset())
    #      - sparseOptFlow motion compensation (GMC): reset() on many
    #        ultralytics builds does NOT touch it, and gmc_method is
    #        sparseOptFlow in botsort_reid.yaml, so its prevFrame would be the
    #        PREVIOUS clip's last frame -> bogus camera motion on frame 1 ->
    #        displaced tracks -> rubbish positions (and rubbish meshes built
    #        from them). We reset it explicitly.
    pred = getattr(model, "predictor", None)
    trackers = getattr(pred, "trackers", None) if pred is not None else None
    if not trackers:
        print("  [tracker] no live tracker yet - fresh start")
        return
    for t in trackers:
        try:
            if hasattr(t, "reset"):
                t.reset()
        except Exception as e:
            print(f"  [tracker] reset() warning: {e}")
        # belt-and-suspenders explicit clear (covers builds where reset() is
        # missing or only partial)
        for attr in ("tracked_stracks", "lost_stracks", "removed_stracks"):
            if hasattr(t, attr):
                try:
                    setattr(t, attr, [])
                except Exception:
                    pass
        if hasattr(t, "frame_id"):
            try:
                t.frame_id = 0
            except Exception:
                pass
        gmc = getattr(t, "gmc", None)
        if gmc is not None and hasattr(gmc, "reset_params"):
            try:
                gmc.reset_params()
            except Exception as e:
                print(f"  [tracker] gmc reset warning: {e}")
    print(f"  [tracker] reset {len(trackers)} live tracker(s) for new job")


def process_video(video_path: str, on_progress=None) -> dict:
    if _RUNTIME_OK:
        print_banner(DEVICE)

    player_model, pitch_homo, pose_model = get_models()

    # Reset ALL per-clip state held on the shared singletons so a new upload
    # starts completely clean: the BoT-SORT tracker (tracks, ReID, motion
    # compensation, ID counter) and the homography calibrator. team/kalman are
    # already created fresh below. This is what stops a previous clip's results
    # from contaminating this one (the rubbish-on-2nd-video problem).
    _reset_tracker(player_model)
    if hasattr(pitch_homo, "reset"):
        pitch_homo.reset()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  {width}x{height}  {fps:.1f}fps  {total_frames} frames")
    print(f"  Processing every {PROCESS_EVERY_N} frame(s)")

    frames_data   = []
    raw_frame_num = 0
    proc_idx      = 0

    team   = TeamAssigner(pitch_length=PITCH_LENGTH)
    kalman = PitchKalman(fps / PROCESS_EVERY_N)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frame_num += 1

        if raw_frame_num % PROCESS_EVERY_N != 0:
            continue

        proc_idx += 1

        frame_data = _analyze_frame(
            frame, raw_frame_num, proc_idx, fps,
            (player_model, pitch_homo, pose_model),
            # persist=True keeps tracking continuous across frames within this
            # clip. Cross-job cleanup is handled by _reset_tracker() above, not
            # by toggling persist (the first .track() call binds the persist
            # value for the whole process, so flipping it mid-run is unsafe).
            team, kalman, persist=True,
        )
        frames_data.append(frame_data)

        _emit_progress(on_progress, raw_frame_num, total_frames)

    cap.release()

    return _finalize_result(frames_data, team, total_frames, fps,
                            width, height, is_image=False)


# ── Image input support ───────────────────────────────────────────────────────
# The client also needs single still-image input. Detection, pitch homography,
# team assignment and pose are all per-frame operations, so they run on one
# image exactly as they run on one video frame. The only real difference is
# downstream in the mesh stage (lift_gvhmr.py): GVHMR is a *temporal* model, so
# for a still image it replicates the frame into a short static clip. Here we
# just emit a one-frame result JSON with the same schema the rest of the app
# already expects.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _analyze_frame(frame, raw_frame_num, proc_idx, fps, models, team, kalman,
                   persist=True):
    """Full per-frame analysis (homography + detection/tracking + team
    observation + pose) on one BGR frame, returning its result dict.

    Shared by process_video (persist=True, tracking carried across frames) and
    process_image (persist=False, a standalone frame). Behaviour for the video
    path is identical to the original inline loop body.
    """
    player_model, pitch_homo, pose_model = models
    from pose import pad_bbox as _pad_bbox

    # Pitch keypoints + homography
    H, fresh, kpt_detections = pitch_homo.get_homography(frame)

    # Store line segments detected by PnLCalib this frame — no re-run needed
    # at export time; video_export.py reads these directly from the JSON.
    line_detections = []
    if HOMOGRAPHY_BACKEND == "pnlcalib" and hasattr(pitch_homo, "overlay_lines"):
        line_detections = pitch_homo.overlay_lines()

    # Detection + tracking — ALWAYS runs
    results = player_model.track(
        frame,
        classes = PERSON_CLASSES + ([CLS_BALL] if TRACK_BALL else []),
        conf    = DET_CONF,
        imgsz   = PLAYER_IMGSZ,
        tracker = TRACKER_CFG,
        device  = DEVICE,
        persist = persist,
        verbose = False
    )[0]

    # ── First pass: walk detections once, build per-player entries and
    # queue up the per-frame batches for ViTPose and team observation.
    players            = []   # final list, kpts filled by second pass
    ball               = None
    team_observations  = []   # (tid, kind, [x1,y1,x2,y2], pitch_x_or_None)
    pose_jobs          = []   # (index_into_players, pbox)

    if results.boxes is not None:
        ih, iw = frame.shape[:2]
        for box in results.boxes:
            cls = int(box.cls)
            if cls not in KIND:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            tid  = int(box.id) if box.id is not None else -1
            conf = float(box.conf)

            if cls == CLS_BALL:
                if conf < BALL_MIN_CONF:
                    continue
                bbox_r    = [round(x1), round(y1), round(x2), round(y2)]
                pitch_pos = None
                if fresh:
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    bx, by = pitch_homo.pixel_to_pitch(cx, cy, H)
                    if bx is not None and -5 <= bx <= PITCH_LENGTH + 5 and -5 <= by <= PITCH_WIDTH + 5:
                        pitch_pos = [round(bx, 2), round(by, 2)]
                if ball is None or conf > ball["conf"]:
                    ball = {"conf": round(conf, 3), "bbox": bbox_r, "pitch_pos": pitch_pos}
                continue

            if conf < PERSON_MIN_CONF:
                continue

            pitch_pos = None
            snap      = False
            if fresh:
                foot_x = (x1 + x2) / 2.0
                foot_y = y2
                px, py = pitch_homo.pixel_to_pitch(foot_x, foot_y, H)
                if px is not None and -5 <= px <= PITCH_LENGTH + 5 and -5 <= py <= PITCH_WIDTH + 5:
                    sx, sy, snap = kalman.update(tid, px, py, proc_idx)
                    pitch_pos = [round(sx, 2), round(sy, 2)]

            kind = KIND[cls]
            if kind in ("player", "goalkeeper"):
                team_observations.append((
                    tid, kind, [x1, y1, x2, y2],
                    pitch_pos[0] if pitch_pos else None,
                ))
            elif kind == "referee":
                # Referees never enter team clustering, but we sample their
                # torso colour so the frontend can show their real kit colour
                # instead of a hardcoded yellow.
                team.observe_referee([x1, y1, x2, y2], frame)

            entry = {
                "id"       : tid,
                "cls"      : cls,
                "kind"     : kind,
                "conf"     : round(conf, 3),
                "bbox"     : [round(x1), round(y1), round(x2), round(y2)],
                "pitch_pos": pitch_pos,
                "kpts"     : None,
            }
            if snap:
                entry["snap"] = True

            # Queue pose work for this entry, don't run yet
            if pose_model is not None and cls in POSE_CLASSES \
                    and (y2 - y1) >= POSE_MIN_BBOX_H \
                    and (proc_idx % POSE_EVERY_N == 0):
                pbox = _pad_bbox(x1, y1, x2, y2, iw, ih)
                pose_jobs.append((len(players), pbox))

            players.append(entry)

    # ── Bbox-collision detection. Flag any pair of player/goalkeeper tracks
    # whose bboxes overlap by more than TEAM_COLLISION_IOU. This must run
    # BEFORE observe_batch() so the team assigner sees the collision flags
    # before deciding whether to (re-)embed the affected tracks this frame.
    _flag_team_collisions(team_observations, team, proc_idx)

    # ── Batched team observation: one ResNet pass for the whole frame's
    # qualifying player/goalkeeper detections.
    if team_observations:
        team.observe_batch(team_observations, frame, proc_idx)

    # ── Batched pose: one ViTPose forward for all qualifying bboxes in this
    # frame, then write kpts back into the corresponding entries.
    if pose_jobs and pose_model is not None:
        pboxes = [pj[1] for pj in pose_jobs]
        pose_results = pose_model.predict_batch(frame, pboxes)
        for (pi, _), (k, s) in zip(pose_jobs, pose_results):
            if k is not None:
                players[pi]["kpts"] = [
                    [round(float(k[j][0]), 1),
                     round(float(k[j][1]), 1),
                     round(float(s[j]), 3)]
                    for j in range(17)
                ]

    return {
        "frame_idx"            : raw_frame_num,
        "timestamp"            : round(raw_frame_num / fps, 3) if fps else 0.0,
        "homography_available" : fresh,
        "pitch_keypoints"      : kpt_detections,
        "pitch_lines"          : line_detections,
        "players"              : players,
        "ball"                 : ball,
    }


def _finalize_result(frames_data, team, total_frames, fps, width, height,
                     is_image=False):
    """Lock teams, relabel stray goalkeepers and assemble the result dict.
    Shared by process_video and process_image so both emit identical schema.
    """
    # Final pass: lock teams
    team.finalize()

    gk_x = {}
    for f in frames_data:
        for p in f["players"]:
            if p["kind"] == "goalkeeper" and p["id"] >= 0 and p["pitch_pos"]:
                gk_x.setdefault(p["id"], []).append(p["pitch_pos"][0])
    gk_demote = set()
    for tid, xs in gk_x.items():
        mx = float(np.median(xs))
        if GK_ZONE_FRAC * PITCH_LENGTH < mx < (1 - GK_ZONE_FRAC) * PITCH_LENGTH:
            gk_demote.add(tid)
    if gk_demote:
        print(f"  Goalkeeper tracks relabelled player (outside goal zone): {sorted(gk_demote)}")

    for f in frames_data:
        for p in f["players"]:
            if p["id"] in gk_demote and p["kind"] == "goalkeeper":
                p["kind"] = "player"
                p["cls"]  = CLS_PLAYER
            if p["kind"] == "referee":
                p["team"] = None
            else:
                p["team"] = team.team_of(p["id"])

    homo_frames    = sum(1 for f in frames_data if f["homography_available"])
    players_frames = sum(1 for f in frames_data if f["players"])
    mapped_players = sum(
        sum(1 for p in f["players"] if p["pitch_pos"] is not None)
        for f in frames_data
    )

    kind_counts = {}
    ball_frames = sum(1 for f in frames_data if f["ball"])
    for f in frames_data:
        for p in f["players"]:
            kind_counts[p["kind"]] = kind_counts.get(p["kind"], 0) + 1
    print(f"  Detections by kind: {kind_counts}  ball frames: {ball_frames}")
    print(f"  Done — {len(frames_data)} frames processed")
    print(f"  Homography OK     : {homo_frames}/{len(frames_data)} frames")
    print(f"  Frames w/ players : {players_frames}")
    print(f"  Mapped players    : {mapped_players}")

    safe_fps = fps if (fps and fps > 0) else 1.0

    return {
        "metadata": {
            "total_frames"    : total_frames,
            "processed_frames": len(frames_data),
            "fps"             : round(fps, 2),
            "playback_fps"    : round(fps / PROCESS_EVERY_N, 2) if fps else 0,
            "duration"        : round(total_frames / safe_fps, 2),
            "width"           : width,
            "height"          : height,
            "pitch_width"     : PITCH_LENGTH,
            "pitch_height"    : PITCH_WIDTH,
            "is_image"        : is_image,
            "input_kind"      : "image" if is_image else "video",
            "teams"           : _team_meta(team),
            "stats": {
                "homography_frames": homo_frames,
                "players_frames"   : players_frames,
                "mapped_players"   : mapped_players,
            },
        },
        "frames": frames_data,
    }


def process_image(image_path: str, on_progress=None) -> dict:
    """Run the pipeline on a single still image. Emits the same result JSON
    schema as process_video, with exactly one frame (frame_idx == 1) and
    metadata["is_image"] == True. The mesh stage (lift_gvhmr.py) detects the
    image input from its extension and replicates the frame into a short
    static clip so GVHMR — a temporal model — can still produce a mesh.
    """
    if _RUNTIME_OK:
        print_banner(DEVICE)

    models = get_models()
    # Reset all per-clip singleton state (tracker + homography) so this upload
    # starts clean, exactly like process_video.
    _reset_tracker(models[0])
    if hasattr(models[1], "reset"):
        models[1].reset()

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot open image: {image_path}")

    height, width = frame.shape[:2]
    print(f"  {width}x{height}  single image")

    if on_progress:
        on_progress({"frame": 0, "total": 1, "percent": 1,
                     "stage": "Analysing image…"})

    team   = TeamAssigner(pitch_length=PITCH_LENGTH)
    kalman = PitchKalman(1.0)   # single observation; fps is irrelevant here

    # persist=True everywhere (image included) so the very first .track() call
    # in the process binds persist=True for BoT-SORT; per-job cleanup is done by
    # _reset_tracker() above. raw_frame_num=1 because an image is processed
    # unconditionally — there is no PROCESS_EVERY_N frame skipping.
    frame_data = _analyze_frame(
        frame, raw_frame_num=1, proc_idx=1, fps=0,
        models=models, team=team, kalman=kalman, persist=True,
    )
    frames_data = [frame_data]

    if on_progress:
        on_progress({"frame": 1, "total": 1, "percent": 90,
                     "stage": "Assigning teams…"})

    result = _finalize_result(frames_data, team, total_frames=1, fps=0,
                              width=width, height=height, is_image=True)

    if on_progress:
        on_progress({"frame": 1, "total": 1, "percent": 100, "stage": "Done"})
    return result


def _team_meta(team):
    kit = None
    if team.team_colors is not None:
        kit = ["#%02x%02x%02x" % (int(round(c[2])), int(round(c[1])), int(round(c[0])))
               for c in team.team_colors]
    ref = None
    rc = getattr(team, "referee_color", None)
    if rc is not None:
        ref = "#%02x%02x%02x" % (int(round(rc[2])), int(round(rc[1])), int(round(rc[0])))
    if kit is None and ref is None:
        return None
    return {"kit_colors": kit, "referee_color": ref}


def _emit_progress(on_progress, raw_frame_num, total_frames, proc_idx=None):
    if on_progress:
        # Emit every processed frame — the frontend polls at 1.2s so the
        # user always sees smooth, up-to-date numbers.
        on_progress({
            "frame"  : raw_frame_num,
            "total"  : total_frames,
            "percent": round(raw_frame_num / total_frames * 100, 1),
            "stage"  : f"Frame {raw_frame_num} / {total_frames}",
        })