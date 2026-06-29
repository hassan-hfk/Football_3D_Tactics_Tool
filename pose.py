"""
pose.py — ViTPose-B (COCO, ONNX) wrapper for the football pipeline.

This mirrors the preprocessing and heatmap decoding from the standalone
Body_Pose_Estimation/pose_pipeline.py so the keypoints produced here match
what you already tested. It runs via ONNX Runtime.

Output per person: 17 COCO keypoints in ORIGINAL FRAME pixel coordinates,
each as [x, y, confidence].

API:
  predict_one(frame, bbox)     -> (kpts(17,2), scores(17,))
  predict_batch(frame, bboxes) -> list of (kpts(17,2), scores(17,)) tuples,
                                  same length and order as `bboxes`.

predict_batch stacks all crops into one (N, 3, 256, 192) tensor and runs a
single session.run call. Mathematically identical to N separate predict_one
calls because ONNX matmul is order-independent at this granularity, but
4-6x faster on CPU and CoreML and 2-3x faster on CUDA, because the
session.run launch and IO overhead are amortized over N crops.

predict_one is now implemented in terms of predict_batch with N=1 so the two
paths cannot drift.

COCO-17 order:
 0 nose 1 eyeL 2 eyeR 3 earL 4 earR 5 shoulderL 6 shoulderR 7 elbowL 8 elbowR
 9 wristL 10 wristR 11 hipL 12 hipR 13 kneeL 14 kneeR 15 ankleL 16 ankleR
"""

import numpy as np

VP_W, VP_H = 192, 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BBOX_PAD = 1.25


def pad_bbox(x1, y1, x2, y2, img_w, img_h, pad=BBOX_PAD):
    """Expand by pad, force 3:4 (W:H) aspect ratio, clamp to image."""
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * pad, (y2 - y1) * pad
    ratio = VP_W / VP_H
    if bw / (bh + 1e-6) > ratio:
        bh = bw / ratio
    else:
        bw = bh * ratio
    return (max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)),
            min(img_w, int(cx + bw / 2)), min(img_h, int(cy + bh / 2)))


class ViTPose:
    def __init__(self, onnx_path: str, device: str = "cpu"):
        # Prefer the centralized session builder from runtime.py if it's
        # importable, else fall back to direct onnxruntime construction so
        # this module stays usable standalone (e.g. in test scripts).
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
        print(f"  ViTPose loaded: {onnx_path} (providers: {providers})")

    # ── Single-bbox path: thin wrapper over predict_batch ─────────────────
    def predict_one(self, frame, bbox):
        """Run pose on one padded bbox. Returns (kpts(17,2) frame px, scores(17,))."""
        out = self.predict_batch(frame, [bbox])
        return out[0]

    # ── Batched path: one session.run for many bboxes ─────────────────────
    def predict_batch(self, frame, bboxes):
        """Run pose on multiple padded bboxes from one frame in a single
        batched session.run call.

        Args:
            frame:  BGR image (full source frame)
            bboxes: list of (nx1, ny1, nx2, ny2) tuples from pad_bbox(...)

        Returns:
            list of (kpts(17,2), scores(17,)) tuples, same length and order
            as `bboxes`. For bboxes too small to crop (cw<=1 or ch<=1) the
            entry is (None, zeros(17)) — identical to predict_one's behaviour.
        """
        import cv2

        out = [(None, np.zeros(17, np.float32))] * len(bboxes)
        if not bboxes:
            return out

        blobs   = []
        centers = []
        scales  = []
        valid_idx = []

        for i, bbox in enumerate(bboxes):
            nx1, ny1, nx2, ny2 = bbox
            cw, ch = nx2 - nx1, ny2 - ny1
            if cw <= 1 or ch <= 1:
                continue
            crop = cv2.resize(frame[ny1:ny2, nx1:nx2], (VP_W, VP_H),
                              interpolation=cv2.INTER_LINEAR)
            blob = crop[:, :, ::-1].astype(np.float32) / 255.0
            blob = (blob - MEAN) / STD
            blob = blob.transpose(2, 0, 1)  # (3, 256, 192)
            blobs.append(blob)
            centers.append(np.array([(nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0]))
            scales.append(np.array([cw, ch], dtype=np.float32))
            valid_idx.append(i)

        if not blobs:
            return out

        batch = np.stack(blobs).astype(np.float32)             # (N, 3, 256, 192)
        heatmaps = self.session.run(
            None, {self.input_name: batch}
        )[0]                                                   # (N, 17, H, W)

        for j, i in enumerate(valid_idx):
            k, s = self._decode(heatmaps[j], centers[j], scales[j])
            out[i] = (k, s)
        return out

    @staticmethod
    def _decode(heatmap, center, scale):
        n, hm_h, hm_w = heatmap.shape
        kpts   = np.zeros((n, 2), np.float32)
        scores = np.zeros(n, np.float32)
        for j in range(n):
            hm = heatmap[j]
            scores[j] = float(hm.max())
            idx = int(hm.argmax()); py, px = idx // hm_w, idx % hm_w
            fpx, fpy = float(px), float(py)
            if 1 <= px < hm_w - 1:
                fpx += 0.25 * np.sign(hm[py, px + 1] - hm[py, px - 1])
            if 1 <= py < hm_h - 1:
                fpy += 0.25 * np.sign(hm[py + 1, px] - hm[py - 1, px])
            kpts[j, 0] = center[0] - scale[0] / 2.0 + (fpx / hm_w) * scale[0]
            kpts[j, 1] = center[1] - scale[1] / 2.0 + (fpy / hm_h) * scale[1]
        return kpts, scores