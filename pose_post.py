"""
pose_post.py — cleans up 2D keypoints before MotionBERT and 3D joints after.

Why the figures move abnormally: ViTPose keypoints on 40-60px tall player
crops jitter by several pixels per frame (measured ~3.5px mean, 7% of body
height). MotionBERT lifts that noise into 3D, where it shows up as limbs that
flail and bones that stretch and shrink (measured bone length variation of
±46% on real output). Nothing in the old pipeline smoothed any of it.

Three stages fix it:
  2D (before lifting)
    - low-confidence joints are interpolated in time from confident frames,
      instead of feeding garbage coordinates into the lifter
    - per-joint Savitzky-Golay smoothing (offline, so a centred window is
      fine: smooths noise without lagging like a causal filter would)
    - optional 2x temporal interpolation back to full video frame rate,
      because the pipeline processes every 2nd frame and MotionBERT was
      trained on full-rate motion
  3D (after lifting)
    - per-joint Savitzky-Golay smoothing
    - constant bone lengths: median bone length over the whole sequence is
      enforced every frame by walking the skeleton from the root, keeping
      joint directions but fixing segment lengths (kills rubber limbs)

All functions take (T, 17, C) numpy arrays. H36M-17 joint order.
"""

import numpy as np
from scipy.signal import savgol_filter

# H36M-17 skeleton as (parent, child) walked root-outward
H36M_BONES = [
    (0, 1), (1, 2), (2, 3),        # right leg: hip->knee->ankle
    (0, 4), (4, 5), (5, 6),        # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),   # spine->thorax->neck->head
    (8, 11), (11, 12), (12, 13),   # left arm
    (8, 14), (14, 15), (15, 16),   # right arm
]

KP_CONF_MIN   = 0.30   # 2D joints below this are treated as missing
SG_WINDOW_2D  = 9      # Savitzky-Golay window (processed frames), odd
SG_WINDOW_3D  = 11
SG_POLY       = 2


def _sg(x, window, poly=SG_POLY):
    """Savitzky-Golay along axis 0, window clamped to sequence length."""
    T = x.shape[0]
    w = min(window, T if T % 2 == 1 else T - 1)
    if w < poly + 2:
        return x
    return savgol_filter(x, w, poly, axis=0)


def interpolate_low_conf(kpts):
    """
    kpts: (T, 17, 3) [x, y, conf]. For every joint, frames with conf below
    KP_CONF_MIN get their XY linearly interpolated from the nearest confident
    frames (held at the edges). Confidence values are kept as they are, so the
    lifter still knows which joints were weak.
    Returns a copy.
    """
    out = kpts.copy()
    T = kpts.shape[0]
    if T < 2:
        return out
    t = np.arange(T)
    for j in range(17):
        good = kpts[:, j, 2] >= KP_CONF_MIN
        if good.all() or not good.any():
            continue
        for c in range(2):
            out[~good, j, c] = np.interp(t[~good], t[good], kpts[good, j, c])
    return out


def smooth_2d(kpts):
    """kpts: (T, 17, 3). Savitzky-Golay on XY only. Returns a copy."""
    out = kpts.copy()
    if kpts.shape[0] >= 5:
        out[:, :, :2] = _sg(kpts[:, :, :2], SG_WINDOW_2D)
    return out


def upsample_2x(kpts):
    """
    (T, 17, 3) -> (2T-1, 17, 3): linear interpolation between consecutive
    processed frames, restoring full video frame rate for MotionBERT.
    Confidence of inserted frames = min of the two neighbours.
    """
    T = kpts.shape[0]
    if T < 2:
        return kpts
    out = np.zeros((2 * T - 1, 17, 3), dtype=kpts.dtype)
    out[0::2] = kpts
    mid = (kpts[:-1] + kpts[1:]) * 0.5
    mid[:, :, 2] = np.minimum(kpts[:-1, :, 2], kpts[1:, :, 2])
    out[1::2] = mid
    return out


def downsample_2x(seq):
    """Inverse of upsample_2x on the lifted output: take every 2nd frame."""
    return seq[0::2]


def smooth_3d(joints):
    """joints: (T, 17, 3) lifted poses. Savitzky-Golay all channels."""
    if joints.shape[0] < 5:
        return joints.copy()
    return _sg(joints, SG_WINDOW_3D)


def enforce_bone_lengths(joints, lengths=None):
    """
    joints: (T, 17, 3). Set every bone to its median length over the sequence
    (or to the provided lengths dict), preserving directions, walking the
    skeleton root-outward. Root joint trajectory is untouched.
    Returns (fixed joints, lengths dict).
    """
    out = joints.copy()
    if lengths is None:
        lengths = {}
        for p, c in H36M_BONES:
            d = np.linalg.norm(joints[:, c] - joints[:, p], axis=1)
            lengths[(p, c)] = float(np.median(d))
    for p, c in H36M_BONES:           # ordered root-outward, parents first
        vec  = out[:, c] - out[:, p]
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm = np.where(norm < 1e-8, 1.0, norm)
        out[:, c] = out[:, p] + vec / norm * lengths[(p, c)]
    return out, lengths


UPRIGHT_MAX_DEG = 35.0   # cap on the upright correction so a genuinely
                         # horizontal run (slide, dive) is not forced vertical


def upright_correct(joints):
    """
    joints: (T, 17, 3) in MotionBERT camera space (Y down). Players inherit
    the broadcast camera's pitch as a permanent forward lean, and noisy small
    crops add a crouch bias. This rotates the WHOLE RUN by one fixed rotation
    so the run-median spine direction (hip->thorax) becomes vertical, rotating
    each frame around its own root joint. Real motion within the run (bending,
    jumping, kicking) is preserved because the correction is constant.
    """
    spine = joints[:, 8] - joints[:, 0]               # (T, 3)
    med   = np.median(spine, axis=0)
    n     = np.linalg.norm(med)
    if n < 1e-6:
        return joints
    med    = med / n
    target = np.array([0.0, -1.0, 0.0])               # up = -Y in camera space

    c = float(np.clip(np.dot(med, target), -1.0, 1.0))
    ang = np.arccos(c)
    if ang < 1e-4:
        return joints
    ang  = min(ang, np.radians(UPRIGHT_MAX_DEG))      # cap the correction
    axis = np.cross(med, target)
    an   = np.linalg.norm(axis)
    if an < 1e-8:
        return joints
    axis = axis / an

    # Rodrigues rotation matrix
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)

    root = joints[:, 0:1]                             # (T, 1, 3)
    return root + (joints - root) @ R.T


def clean_2d_sequence(kpts, upsample=True):
    """Full 2D pre-lift treatment. kpts (T,17,3) -> (T or 2T-1, 17, 3)."""
    x = interpolate_low_conf(kpts)
    x = smooth_2d(x)
    if upsample:
        x = upsample_2x(x)
    return x


def clean_3d_sequence(joints, downsample=True):
    """Full 3D post-lift treatment. joints (T,17,3)."""
    x = upright_correct(joints)
    x = smooth_3d(x)
    x, _ = enforce_bone_lengths(x)
    if downsample:
        x = downsample_2x(x)
    return x
