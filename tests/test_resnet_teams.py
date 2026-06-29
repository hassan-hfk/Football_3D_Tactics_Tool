"""
test_resnet_teams.py -- visual test: ResNet-26 embeddings + k-means for team
clustering, drawn as colour-coded bounding boxes on the video.

WHY RESNET-26
  Per Szymon Kulpinski's evaluation ("Clustering Football Players using Image
  Embeddings, UMAP, and K-Means", Mar 2025), ResNet-26 gave the best accuracy
  AND speed trade-off out of 19 embedding models tested (beating SigLIP and
  CLIP variants) for exactly this player-clustering task. This script is the
  ResNet-26 counterpart of test_siglip_teams.py, same pipeline shape, just a
  different embedder.

  Caveat from that article worth remembering: the pipeline (any embedder +
  UMAP + K-Means) was found to be non-deterministic -- repeated runs on the
  same data can give different cluster accuracy (avg ~5% spread, up to ~8% in
  the worst case). If results look "almost right but flipped" or slightly
  different between runs, that's expected; run it a couple of times.

WHAT THIS DOES
  1. Runs the existing player detection model (player_detection_v26s.pt) on a
     video, frame by frame (every Nth frame).
  2. For every player/goalkeeper/referee detection, crops the bbox and embeds
     it with microsoft/resnet-26 (pooled feature vector after global average
     pooling).
  3. Collects embeddings from the first detections (--bootstrap of them),
     reduces with PCA (lightweight stand-in for UMAP, no extra dependency),
     and runs k-means with k=3 (two teams + referee/other).
  4. For each resulting cluster, computes a representative colour by averaging
     the actual torso pixel colours of crops in that cluster (purely for
     VISUALIZATION -- the clustering itself never uses this colour).
  5. Re-runs over the video, drawing each detection's bbox in its cluster's
     representative colour, labelled with cluster id + detector kind
     (player/goalkeeper/referee) + track id.

USAGE
  python test_resnet_teams.py path/to/video.mp4 [--out out.mp4] [--frames 200]
                               [--every 2] [--bootstrap 40] [--k 3]

REQUIREMENTS (in addition to the app's requirements.txt)
  pip install transformers
  (torch, ultralytics, opencv-python, numpy, scikit-learn already required --
  same as test_siglip_teams.py, no new deps beyond transformers)

NOTES
  - This is a STANDALONE diagnostic script, not wired into pipeline.py.
  - ResNet runs on the configured device (mps/cuda/cpu, auto-detected).
  - microsoft/resnet-26 download is small (~50-60MB), much lighter than
    SigLIP's ~400MB -- first run will still need to download it once.
"""

import os
import sys
import argparse
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO


# -- Config -------------------------------------------------------------------
PLAYER_MODEL_PATH = os.path.join("models", "player_detection_v26s.pt")
RESNET_MODEL_NAME = "microsoft/resnet-26"

DET_CONF        = 0.15
PERSON_MIN_CONF = 0.35
PLAYER_IMGSZ    = 1280
TRACKER_CFG     = "botsort_reid.yaml"

# Detector classes (matches pipeline.py)
CLS_BALL, CLS_GOALKEEPER, CLS_PLAYER, CLS_REFEREE = 0, 1, 2, 3
KIND = {CLS_GOALKEEPER: "goalkeeper", CLS_PLAYER: "player", CLS_REFEREE: "referee"}
PERSON_CLASSES = [CLS_GOALKEEPER, CLS_PLAYER, CLS_REFEREE]

# Torso region for the VISUALIZATION colour only (same crop fractions as
# team_assign.py's torso_lab, for an apples-to-apples comparison)
TORSO_TOP, TORSO_BOTTOM = 0.22, 0.58
TORSO_LEFT, TORSO_RIGHT = 0.25, 0.75
MIN_BBOX_H = 28

# Cluster label colours (BGR) used if a cluster's mean torso colour can't be
# computed (e.g. all crops too small) -- fallback only, not the primary signal
FALLBACK_COLORS = [
    (255, 80, 80),    # cluster 0 fallback - blue-ish (BGR)
    (80, 80, 255),    # cluster 1 fallback - red-ish
    (0, 220, 255),    # cluster 2 fallback - yellow
    (180, 80, 255),   # cluster 3+ fallback - pink
]


def _pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def crop_bbox(frame, bbox, pad=0.0):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    ih, iw = frame.shape[:2]
    if pad:
        w, h = x2 - x1, y2 - y1
        x1 -= int(w * pad); x2 += int(w * pad)
        y1 -= int(h * pad); y2 += int(h * pad)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def torso_mean_bgr(frame, bbox):
    """Mean BGR of the torso region, for VISUALIZATION colour only -- not
    used for clustering. Returns None if the crop is unusable."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = y2 - y1, x2 - x1
    if h < MIN_BBOX_H or w < 4:
        return None
    ty1 = y1 + int(h * TORSO_TOP)
    ty2 = y1 + int(h * TORSO_BOTTOM)
    tx1 = x1 + int(w * TORSO_LEFT)
    tx2 = x1 + int(w * TORSO_RIGHT)
    ih, iw = frame.shape[:2]
    ty1, ty2 = max(0, ty1), min(ih, ty2)
    tx1, tx2 = max(0, tx1), min(iw, tx2)
    if ty2 - ty1 < 2 or tx2 - tx1 < 2:
        return None
    crop = frame[ty1:ty2, tx1:tx2]
    return crop.reshape(-1, 3).mean(axis=0)  # BGR


class ResNetEmbedder:
    """Embeds image crops with microsoft/resnet-26, returning the pooled
    (global-average-pooled) feature vector -- a fixed-size embedding per
    crop, same role as SigLIP's pooler_output in test_siglip_teams.py."""

    def __init__(self, model_name, device):
        from transformers import AutoImageProcessor, ResNetModel
        import torch
        self.torch = torch
        self.device = device
        print(f"  Loading ResNet: {model_name} (device={device})")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetModel.from_pretrained(model_name).to(device).eval()

    def embed(self, bgr_crops):
        """bgr_crops: list of HxWx3 BGR numpy arrays. Returns (N, D) embeddings."""
        if not bgr_crops:
            return np.zeros((0, 2048), dtype=np.float32)
        rgb_crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in bgr_crops]
        inputs = self.processor(images=rgb_crops, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model(**inputs)
        # pooler_output: (N, C, 1, 1) -> flatten to (N, C)
        feats = out.pooler_output.squeeze(-1).squeeze(-1).cpu().numpy()
        return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Input video path")
    ap.add_argument("--out", default="resnet_teams_out.mp4", help="Output annotated video")
    ap.add_argument("--frames", type=int, default=200, help="Max processed frames")
    ap.add_argument("--every", type=int, default=2, help="Process every Nth raw frame")
    ap.add_argument("--bootstrap", type=int, default=40, help="Detections to collect before clustering")
    ap.add_argument("--k", type=int, default=3, help="k-means clusters (3 = two teams + referee/other)")
    ap.add_argument("--model", default=RESNET_MODEL_NAME, help="ResNet model name")
    args = ap.parse_args()

    device = _pick_device()
    print(f"  Device: {device}")

    print(f"  Loading detector: {PLAYER_MODEL_PATH}")
    detector = YOLO(PLAYER_MODEL_PATH)

    embedder = ResNetEmbedder(args.model, device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps / args.every, (w, h))

    # -- Pass 1: collect detections + embeddings until bootstrap, fit k-means --
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    print("  Pass 1: collecting bootstrap detections...")
    boot_embeds = []
    boot_colors = []   # torso mean BGR, for visualization colour later
    boot_meta   = []   # (raw_frame_idx, bbox, kind, track_id)

    raw_idx = 0
    proc_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_idx += 1
        if raw_idx % args.every != 0:
            continue
        proc_idx += 1
        if proc_idx > args.frames:
            break

        results = detector.track(
            frame, classes=PERSON_CLASSES, conf=DET_CONF, imgsz=PLAYER_IMGSZ,
            tracker=TRACKER_CFG, persist=True, verbose=False
        )[0]

        if results.boxes is None:
            continue

        crops, metas, colors = [], [], []
        for box in results.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            if conf < PERSON_MIN_CONF:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            tid = int(box.id) if box.id is not None else -1
            crop = crop_bbox(frame, (x1, y1, x2, y2))
            if crop is None or crop.size == 0:
                continue
            crops.append(crop)
            metas.append((raw_idx, (x1, y1, x2, y2), KIND.get(cls, "?"), tid))
            colors.append(torso_mean_bgr(frame, (x1, y1, x2, y2)))

        if crops:
            feats = embedder.embed(crops)
            for f, m, c in zip(feats, metas, colors):
                boot_embeds.append(f)
                boot_meta.append(m)
                boot_colors.append(c)

        if len(boot_embeds) >= args.bootstrap:
            break

    if len(boot_embeds) < max(args.k, 4):
        sys.exit(f"Not enough detections to cluster ({len(boot_embeds)} found, "
                 f"need at least {max(args.k,4)}). Try a longer video or lower --bootstrap.")

    X = np.stack(boot_embeds)
    print(f"  Collected {len(X)} bootstrap embeddings (dim={X.shape[1]})")

    # PCA to a small number of dims keeps k-means stable on ResNet's
    # high-dimensional output without needing the umap dependency.
    n_comp = min(16, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    Xp = pca.fit_transform(X)

    km = KMeans(n_clusters=args.k, n_init=10, random_state=0)
    labels = km.fit_predict(Xp)
    print(f"  k-means cluster sizes: {np.bincount(labels).tolist()}")

    # Representative colour per cluster = mean torso BGR of its members
    cluster_color = {}
    for c in range(args.k):
        cols = [boot_colors[i] for i in range(len(labels))
                if labels[i] == c and boot_colors[i] is not None]
        if cols:
            cluster_color[c] = tuple(int(v) for v in np.mean(cols, axis=0))
        else:
            cluster_color[c] = FALLBACK_COLORS[c % len(FALLBACK_COLORS)]

    print("  Cluster representative colours (BGR):")
    for c, col in cluster_color.items():
        kinds_in_cluster = [boot_meta[i][2] for i in range(len(labels)) if labels[i] == c]
        track_ids = set(boot_meta[i][3] for i in range(len(labels)) if labels[i] == c)
        print(f"    cluster {c}: color={col}  n={int(np.sum(labels==c))}  "
              f"tracks={len(track_ids)}  kinds={Counter(kinds_in_cluster)}")

    # -- Pass 2: replay the whole clip, classify every detection live, draw ----
    print("  Pass 2: annotating video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    raw_idx = 0
    proc_idx = 0

    # Track-level cluster vote so a track's box colour doesn't flicker frame
    # to frame: majority vote over the track's history so far.
    track_votes = {}  # tid -> Counter(cluster -> count)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_idx += 1
        if raw_idx % args.every != 0:
            continue
        proc_idx += 1
        if proc_idx > args.frames:
            break

        results = detector.track(
            frame, classes=PERSON_CLASSES + [CLS_BALL], conf=DET_CONF, imgsz=PLAYER_IMGSZ,
            tracker=TRACKER_CFG, persist=True, verbose=False
        )[0]

        out_frame = frame.copy()

        if results.boxes is not None:
            crops, metas = [], []
            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                tid = int(box.id) if box.id is not None else -1

                if cls == CLS_BALL:
                    cv2.rectangle(out_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                   (255, 255, 255), 1)
                    continue
                if conf < PERSON_MIN_CONF:
                    continue
                crop = crop_bbox(frame, (x1, y1, x2, y2))
                if crop is None or crop.size == 0:
                    continue
                crops.append(crop)
                metas.append((x1, y1, x2, y2, cls, tid))

            if crops:
                feats = embedder.embed(crops)
                feats_p = pca.transform(feats)
                cluster_ids = km.predict(feats_p)

                for (x1, y1, x2, y2, cls, tid), c in zip(metas, cluster_ids):
                    if tid >= 0:
                        votes = track_votes.setdefault(tid, Counter())
                        votes[int(c)] += 1
                        c_disp = votes.most_common(1)[0][0]
                    else:
                        c_disp = int(c)

                    color = cluster_color.get(c_disp, FALLBACK_COLORS[0])
                    kind = KIND.get(cls, "?")
                    label = f"{kind[:3]}#{tid} c{c_disp}"

                    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                    cv2.rectangle(out_frame, p1, p2, color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(out_frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
                    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
                    cv2.putText(out_frame, label, (p1[0] + 2, p1[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        writer.write(out_frame)

    writer.release()
    cap.release()
    print(f"\nDone. Output: {args.out}")


if __name__ == "__main__":
    main()
