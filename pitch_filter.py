"""
pitch_filter.py — per-track constant-velocity Kalman filter for pitch positions.

Replaces the exponential lerp smoothing. What it does better:
  - smooths foot-point jitter without lagging behind real movement (the lerp
    always lags, the Kalman tracks velocity so it leads correctly)
  - coasts a player through 1-2 missed detections instead of leaving a hole
  - rejects physically impossible jumps (homography glitch frames) with a
    measurement gate, so one bad frame cannot teleport a player

All units are pitch meters. The time step is one PROCESSED frame.

Tuning (the two numbers that matter):
  Q_POS_STD / Q_VEL_STD  how much we let true motion change per frame.
                         Bigger = more responsive, less smooth.
  R_STD                  how noisy we believe a single measurement is.
                         Bigger = smoother, laggier.
With 12.5 processed fps, a sprinting player moves ~0.6 m/frame and direction
changes are gradual, while the measured foot point jitters around ±0.3-0.5 m.
"""

import numpy as np

Q_POS_STD   = 0.05     # m / frame   process noise on position
Q_VEL_STD   = 0.10     # m / frame^2 process noise on velocity
R_STD       = 0.90     # m           measurement noise
GATE_M      = 4.0      # reject a measurement further than this from prediction
GATE_RELAX  = 2        # after this many consecutive rejections, accept anyway
                       # (the player really did move, e.g. tracker re-lock)
MAX_COAST   = 2        # predict through at most this many missing frames
MAX_SPEED   = 11.0     # m/s hard cap on filter velocity (~world record sprint)


class _Track:
    __slots__ = ("x", "P", "last_idx", "rejects")

    def __init__(self, px, py, idx):
        # state [x, y, vx, vy]
        self.x = np.array([px, py, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([1.0, 1.0, 4.0, 4.0])
        self.last_idx = idx
        self.rejects  = 0


class PitchKalman:
    """
    update(tid, px, py, proc_idx) -> (sx, sy) filtered position
    predict_missing(proc_idx)     -> {tid: (x, y)} coasted positions for tracks
                                     not seen this frame (within MAX_COAST)
    """

    def __init__(self, fps_processed: float):
        self.tracks = {}
        dt  = 1.0
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)
        q_p, q_v = Q_POS_STD ** 2, Q_VEL_STD ** 2
        self.Q = np.diag([q_p, q_p, q_v, q_v])
        self.R = np.eye(2) * R_STD ** 2
        self.max_step = MAX_SPEED / fps_processed   # m per processed frame

    def update(self, tid, px, py, proc_idx):
        """Returns (sx, sy, snap). snap=True means the position jumped for real
        (new track, tracker re-lock after a gap, or confirmed teleport) and the
        frontend should hard-set the figure there instead of gliding to it."""
        if tid < 0:
            return px, py, True                 # untracked: pass through

        tr = self.tracks.get(tid)
        gap = (proc_idx - tr.last_idx) if tr is not None else None

        if tr is None or gap > MAX_COAST + 1:
            # new track, or came back after a long gap (replay / re-entry):
            # restart the filter, never glide across the gap
            self.tracks[tid] = _Track(px, py, proc_idx)
            return px, py, True

        # Predict forward over the gap (usually 1 step)
        for _ in range(gap):
            tr.x = self.F @ tr.x
            tr.P = self.F @ tr.P @ self.F.T + self.Q

        # Gate: physically impossible jump = homography glitch, reject it
        z    = np.array([px, py], dtype=np.float64)
        pred = self.H @ tr.x
        if np.linalg.norm(z - pred) > max(GATE_M, self.max_step * (gap + 2)):
            tr.rejects += 1
            if tr.rejects <= GATE_RELAX:
                tr.last_idx = proc_idx          # hold position this frame
                return float(pred[0]), float(pred[1]), False
            # too many rejections in a row: the jump is real, re-init there
            self.tracks[tid] = _Track(px, py, proc_idx)
            return px, py, True
        tr.rejects = 0

        # Standard KF update
        S = self.H @ tr.P @ self.H.T + self.R
        K = tr.P @ self.H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (z - pred)
        tr.P = (np.eye(4) - K @ self.H) @ tr.P

        # Velocity sanity cap
        v = tr.x[2:]
        speed = np.linalg.norm(v)
        if speed > self.max_step:
            tr.x[2:] = v / speed * self.max_step

        tr.last_idx = proc_idx
        return float(tr.x[0]), float(tr.x[1]), False

    def predict_missing(self, proc_idx):
        """Coasted positions for tracks not updated this frame."""
        out = {}
        for tid, tr in self.tracks.items():
            gap = proc_idx - tr.last_idx
            if 1 <= gap <= MAX_COAST:
                x = tr.x.copy()
                for _ in range(gap):
                    x = self.F @ x
                out[tid] = (float(x[0]), float(x[1]))
        return out
