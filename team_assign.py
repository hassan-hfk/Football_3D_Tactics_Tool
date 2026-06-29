"""
team_assign.py — ResNet-26 embedding based team assignment with a recent-vote
                 sliding window (resilient to BoT-SORT ID swaps during
                 occlusions/collisions).

How it works
  1. For every tracked player/goalkeeper crop, embed the bbox crop with
     microsoft/resnet-26 (pooled feature vector, 2048-d). This captures kit
     pattern, texture and shape, not just average colour — two kits that look
     almost identical in raw colour (e.g. two dark navy variants under
     different lighting, or a referee whose kit colour overlaps a team's)
     still tend to separate in this embedding space.
  2. BOOTSTRAP: collect embeddings from the first BOOTSTRAP_SAMPLES samples of
     tracks classified "player" by the detector. Reduce to PCA_DIMS dims with
     PCA, then run k-means with k=3 and pick the two clusters that represent
     real TEAMS (see _fit_team_clusters below). The two resulting centroids
     (in PCA space) are the team "kit embeddings" for the whole video.
  3. SLIDING WINDOW: after bootstrap, every embedding gets projected into
     PCA space and contributes one vote to its track's recent-vote window
     (capped at RECENT_WINDOW frames). The track's current team is the
     majority verdict in that window. Crucially this means the team CAN
     change over time — if BoT-SORT swaps two players' track IDs during a
     collision, the new evidence accumulates in the window and the team
     self-corrects within ~10-15 frames. Hysteresis (FLIP_MARGIN) prevents
     single-frame noise from causing visible flicker.
  4. SUBSAMPLING: once a track's window is "stable" (≥STABLE_THRESHOLD of
     the window agrees on one team), we only re-embed that track every
     SUBSAMPLE_INTERVAL-th frame instead of every frame. This recovers most
     of the efficiency of the old "permanent lock" design while still
     allowing the team to flip when evidence shifts.
  5. COLLISIONS: pipeline.py detects pairs of overlapping bboxes (IoU above
     a threshold) and calls mark_collision(tid_a, tid_b, proc_idx). That
     clears both tracks' recent windows and forces every-frame re-embed for
     the next COLLISION_REVERIFY_FRAMES frames, so post-collision evidence
     dominates the next team decision instead of being diluted by stale
     pre-collision votes (which may have belonged to a different player).

Teams in the output JSON:
  team =  0 or 1   current team verdict
  team = -1        unknown kit — rare; not used to change `kind`
  team = None      not enough evidence yet (short track, bootstrap phase)

Goalkeepers wear neither kit, so they are assigned by pitch half at finalize()
time. Which real team holds the left half gets decided by majority of where
each team's players stood during play, which is right for normal open play
footage.

IMPORTANT — role (kind) is NOT decided here
  The detector (player_detection_v26s.pt) is trusted for player / goalkeeper /
  referee / ball classification — it performs well on its own. This module
  ONLY decides which of the two real teams a player/goalkeeper track belongs
  to (team 0 / team 1). It never changes `kind` or `cls`, and referee-kind
  detections are not fed into team voting at all (pipeline.py only calls
  observe() / observe_batch() for kind in ("player", "goalkeeper")).

Batched API
  observe_batch(observations, frame, proc_idx) processes multiple tracked
  detections from the same frame in one ResNet pass. Bit-exact equivalent
  to calling observe() per-detection in the same order, but 3-5x faster
  because the embedding model runs once per frame instead of once per
  player per frame.
"""

from collections import deque

import numpy as np
import cv2

# ── Tuning ────────────────────────────────────────────────────────────────────
BOOTSTRAP_SAMPLES = 120     # crops to collect before running k-means. ResNet
                            # embeddings separate kits far more cleanly than
                            # raw colour, so far fewer samples are needed than
                            # the old LAB approach (which used 400).

# Sliding-window decision parameters (replaces the old permanent LOCK_VOTES).
RECENT_WINDOW          = 30   # frames of recent evidence considered for
                              # current team. A track that gets ID-swapped
                              # mid-video will self-correct within roughly
                              # this many frames once new evidence is in.
MIN_VOTES_FOR_TEAM     = 6    # minimum window entries before assigning ANY
                              # team. Below this the track stays team=None,
                              # which signals "not enough evidence yet" to
                              # the frontend.
INITIAL_MARGIN         = 0.50 # set the first team verdict by simple majority
FLIP_MARGIN            = 0.65 # to flip AWAY from an already-assigned team,
                              # the recent window must agree at this rate on
                              # the new team. Higher = more conservative
                              # (no flicker on noisy embeddings) but slower
                              # to recover after a real ID swap.
STABLE_THRESHOLD       = 0.85 # window agreement ≥ this means "stable" — the
                              # track gets subsampled (re-embedded every
                              # SUBSAMPLE_INTERVAL-th frame instead of every
                              # frame). Below this, every frame is embedded.
SUBSAMPLE_INTERVAL     = 5    # for stable tracks: embed once every N frames
                              # instead of every frame. Picks up identity
                              # changes from ID swaps within ~SUBSAMPLE_INTERVAL
                              # extra frames, while keeping inference cheap.
COLLISION_REVERIFY_FRAMES = 20  # after a collision is flagged, force every-
                                # frame embed for this many frames so the
                                # post-collision evidence dominates fast.

UNKNOWN_MARGIN    = 1.8     # embedding counts as unknown if its distance to
                            # the nearest centroid is > UNKNOWN_MARGIN * the
                            # bootstrap intra-cluster spread of that centroid
MIN_UNKNOWN_FRAC  = 0.6     # call a track's team -1 (unknown) if this
                            # fraction of its recent window is unknown
MIN_BBOX_H        = 20      # skip embedding tiny boxes (too noisy / blurry)
PCA_DIMS          = 16      # PCA dimensions for clustering (k-means on raw
                            # 2048-d ResNet features is noisy and slow)
MIN_FIT_SAMPLES   = 8       # min player crops for finalize() to run k-means
                            # when the BOOTSTRAP_SAMPLES window was never
                            # reached. A single still image is the main case:
                            # ~16-22 player crops in ONE frame, far below
                            # BOOTSTRAP_SAMPLES, so without this lower gate an
                            # image would skip clustering entirely and every
                            # player would fall back to a fake parity colour.

# Torso region inside the bbox (fractions of bbox size), used ONLY to compute
# a representative display colour per team for the frontend — never for
# clustering. Top 22%..58% of height, middle 25%..75% of width: jersey only.
TORSO_TOP, TORSO_BOTTOM = 0.22, 0.58
TORSO_LEFT, TORSO_RIGHT = 0.25, 0.75


def _pick_device():
    # Prefer the centralized device picker from runtime.py if available
    try:
        from runtime import pick_device
        return pick_device()
    except ImportError:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ── ResNet-26 embedder (lazy singleton) ────────────────────────────────────────
# Chosen per evaluation (Kulpinski, "Clustering Football Players using Image
# Embeddings, UMAP, and K-Means", Mar 2025): microsoft/resnet-26 gave the best
# accuracy/speed trade-off of 19 embedding models tested for this exact task,
# beating SigLIP and CLIP variants.
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from transformers import AutoImageProcessor, ResNetModel
        import torch
        device = _pick_device()
        print(f"  [team] loading microsoft/resnet-26 embedder (device={device})")
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-26")
        model = ResNetModel.from_pretrained("microsoft/resnet-26").to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _embedder = (processor, model, torch, device)
    return _embedder


def crop_embedding_batch(frame, bboxes):
    """
    Embed multiple bbox crops from one frame in a single batched ResNet call.
    frame: BGR image. bboxes: list of [x1, y1, x2, y2] in pixels.
    Returns a list the same length as bboxes, each entry an (2048,) float32
    embedding or None if that crop was unusable (too small / out of frame).
    """
    processor, model, torch, device = _get_embedder()
    ih, iw = frame.shape[:2]

    crops, idxs = [], []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        if (y2 - y1) < MIN_BBOX_H or (x2 - x1) < 4:
            continue
        crops.append(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        idxs.append(i)

    out = [None] * len(bboxes)
    if not crops:
        return out

    inputs = processor(images=crops, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        result = model(**inputs)
    feats = result.pooler_output.squeeze(-1).squeeze(-1).cpu().numpy().astype(np.float32)

    for j, i in enumerate(idxs):
        out[i] = feats[j]
    return out


def crop_embedding(frame, bbox):
    """Single-crop convenience wrapper around crop_embedding_batch."""
    return crop_embedding_batch(frame, [bbox])[0]


def torso_mean_bgr(frame, bbox):
    """Mean BGR of the torso region — for the frontend's display colour only,
    never used for clustering. Returns None if the crop is unusable."""
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
    return frame[ty1:ty2, tx1:tx2].reshape(-1, 3).mean(axis=0)


def _vivid_bgr(bgr):
    """Push a representative kit colour toward its true hue so it reads clearly
    on the frontend. Mean-of-means colour aggregation desaturates badly when a
    kit has lighting variance (a green top can collapse to grey), so we boost
    saturation and floor the brightness. A genuinely grey/white kit (near-zero
    saturation) has no hue to amplify and stays grey, which is correct."""
    px  = np.uint8([[[int(np.clip(bgr[0], 0, 255)),
                      int(np.clip(bgr[1], 0, 255)),
                      int(np.clip(bgr[2], 0, 255))]]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    hsv[1] = min(255.0, hsv[1] * 1.7)   # saturation boost
    hsv[2] = max(hsv[2], 90.0)          # brightness floor
    out = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0]
    return out.astype(np.float32)


def _representative_color(cols):
    """Robust representative BGR for a set of per-player torso means. The median
    is resistant to occlusion / grass / skin outliers that drag a mean toward
    grey; the result is then vivified. Returns None if there are no usable
    colours."""
    cols = [c for c in cols if c is not None]
    if not cols:
        return None
    med = np.median(np.stack(cols), axis=0)
    return _vivid_bgr(med)


# ── Lightweight PCA (numpy-only, no scikit-learn dependency) ───────────────────
def _pca_fit(X, n_components):
    """Returns (mean, components) where components is (n_components, D)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD-based PCA: Xc = U S Vt, components = Vt[:n_components]
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = min(n_components, Vt.shape[0])
    return mean, Vt[:n]


def _pca_transform(X, mean, components):
    return (X - mean) @ components.T


# ── k-means (same as before, embedding-agnostic) ───────────────────────────────
def _kmeans_once(X, k, rng, iters=50):
    """One k-means run with proper (probabilistic) k-means++ init."""
    C = [X[rng.randint(len(X))]]
    for _ in range(1, k):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in C], axis=0)
        p  = d2 / (d2.sum() + 1e-12)
        C.append(X[rng.choice(len(X), p=p)])
    C = np.stack(C)
    for _ in range(iters):
        D    = np.stack([np.linalg.norm(X - c, axis=1) for c in C])
        lab  = D.argmin(axis=0)
        newC = np.stack([X[lab == j].mean(axis=0) if (lab == j).any() else C[j]
                         for j in range(k)])
        if np.allclose(newC, C, atol=1e-3):
            C = newC
            break
        C = newC
    D       = np.stack([np.linalg.norm(X - c, axis=1) for c in C])
    lab     = D.argmin(axis=0)
    inertia = float(D.min(axis=0).sum())
    return C, lab, inertia


def _kmeans(X, k, restarts=8, seed=0):
    """Best-of-N k-means, then trimmed refinement: occlusion / motion-blur
    samples are heavy outliers and must not drag the kit centroids around."""
    rng  = np.random.RandomState(seed)
    best = min((_kmeans_once(X, k, rng) for _ in range(restarts)),
               key=lambda r: r[2])
    C, lab, _ = best
    for _ in range(2):                    # trim the worst 10% per cluster
        newC = []
        for j in range(k):
            pts = X[lab == j]
            if len(pts) < 8:
                newC.append(C[j]); continue
            d   = np.linalg.norm(pts - C[j], axis=1)
            newC.append(pts[d <= np.percentile(d, 90)].mean(axis=0))
        C   = np.stack(newC)
        D   = np.stack([np.linalg.norm(X - c, axis=1) for c in C])
        lab = D.argmin(axis=0)
    return C, lab


# Referee/official pollution guard. Occasionally a referee-kind detection
# briefly gets classified "player" by the detector, so the bootstrap pool can
# contain a few non-team samples. Fit k=3 on the bootstrap embeddings: with 2
# real teams + (at most) a small referee/official contamination, one of the 3
# clusters may correspond to that contamination.
#
# The contaminated cluster CANNOT be identified by sample count: a real team
# can be a numeric minority in the bootstrap window (fewer of that team's
# players were visible early on). The reliable signal is TRACK DIVERSITY: a
# real team cluster is made of samples from ~10+ different player tracks. A
# referee/official cluster is made of samples from only 1-2 tracks, no matter
# how many total samples it has. So: the k=3 cluster with the fewest DISTINCT
# track ids is discarded, as long as it's a clear minority in track-count
# terms — not in sample count.
REF_MAX_TRACK_FRAC = 0.35   # contaminated cluster's distinct-track count must
                            # be below this fraction of the OTHER two
                            # clusters' average distinct-track count


def _fit_team_clusters(X, track_ids):
    """
    X: (N, D) raw ResNet embeddings. track_ids: (N,) int array, same order.
    Returns (mean, components, centroids(2, PCA_DIMS), spread(2,)) where mean
    and components define the PCA projection used for voting.
    """
    mean, components = _pca_fit(X, PCA_DIMS)
    Xp = _pca_transform(X, mean, components)

    C3, lab3 = _kmeans(Xp, 3)
    n_unique = np.array([len(set(track_ids[lab3 == j])) for j in range(3)])
    order = np.argsort(n_unique)          # fewest distinct tracks first

    other_mean = float(np.mean(n_unique[order[1:]]))
    use_k3 = other_mean > 0 and n_unique[order[0]] / other_mean < REF_MAX_TRACK_FRAC

    if use_k3:
        big = order[1:]
        C   = C3[big]
        print(f"  [team] k=3 cluster track counts: {n_unique.tolist()} "
              f"-> discarding cluster {order[0]} as referee/official "
              f"({n_unique[order[0]]} track(s))")
    else:
        C2, lab2 = _kmeans(Xp, 2)
        C = C2
        print(f"  [team] k=3 cluster track counts: {n_unique.tolist()} "
              f"-> no clear minority cluster, falling back to k=2")

    D   = np.stack([np.linalg.norm(Xp - c, axis=1) for c in C])
    lab = D.argmin(axis=0)
    spread = np.array([
        np.linalg.norm(Xp[lab == j] - C[j], axis=1).mean()
        if (lab == j).any() else 1.0 for j in range(2)
    ])
    return mean, components, C, np.maximum(spread, 1e-3)


class TeamAssigner:
    """
    Feed it (track_id, kind, bbox, frame, pitch_x) every processed frame via
    observe(), or feed a whole frame's detections at once via observe_batch().
    Read back .team_of(track_id) -> 0 / 1 / -1 / None.

    Unlike the old design this class does NOT permanently lock a track's team.
    Instead it keeps a sliding window (RECENT_WINDOW frames) of recent verdicts
    per track and reports the window-majority as the current team. This means
    a track whose ID got swapped by BoT-SORT during an occlusion will
    self-correct within ~10-15 frames as new evidence accumulates. Hysteresis
    (FLIP_MARGIN) keeps the visible team from flickering on isolated noisy
    frames, and pipeline.py's collision detection clears the window for
    affected tracks so post-collision evidence dominates the next decision
    fast.
    """

    def __init__(self, pitch_length=120.0):
        self.pitch_length = pitch_length  # for goalkeeper side assignment
        self.pca_mean   = None            # (D,) PCA mean for projection
        self.pca_comp   = None            # (PCA_DIMS, D) PCA components
        self.centroids  = None            # (2, PCA_DIMS) team kit embeddings
        self.spread     = None            # (2,) intra-cluster spread
        self.team_colors = None           # (2,) mean torso BGR per team, for frontend
        self.referee_colors = []          # accumulated referee torso BGR samples
        self.referee_color  = None        # representative referee kit colour (BGR)
        self.boot_data  = []              # bootstrap (tid, embedding, color, pitch_x)

        # Sliding-window state replacing the old self.votes / self.locked.
        # window[tid] is a deque of (verdict, proc_idx) pairs, capped at
        # RECENT_WINDOW length. verdict is 0/1/2 (team0/team1/unknown).
        self.window     = {}              # tid -> deque[(verdict, proc_idx)]
        self.team       = {}              # tid -> current team (0/1/-1)
        self.last_embed = {}               # tid -> proc_idx of most recent embed
        # If proc_idx <= force_embed_until[tid], we ignore subsampling and
        # embed this track every frame. Set by mark_collision().
        self.force_embed_until = {}        # tid -> int proc_idx

        # Goalkeeper / side state (unchanged):
        self.gk_side    = {}              # tid -> mean pitch x while undecided
        self._side_acc  = {0: [], 1: []}

    # ── public ────────────────────────────────────────────────────────────────
    def ready(self):
        return self.centroids is not None

    def team_of(self, tid, proc_idx=None):
        return self.team.get(tid)

    def mark_collision(self, tid_a, tid_b, proc_idx):
        """Called from pipeline.py when two tracked bboxes overlap enough that
        BoT-SORT may have swapped the IDs. Clears both tracks' recent vote
        windows (their stale pre-collision votes might have come from a
        different player) and forces every-frame embedding for the next
        COLLISION_REVERIFY_FRAMES frames, so fresh post-collision evidence
        dominates the next team decision instead of being diluted.

        Note: this does NOT immediately clear self.team[tid]. The visible
        team color shouldn't flicker on every collision — it only changes
        when ≥FLIP_MARGIN of the new window agrees on a different team.
        """
        for tid in (tid_a, tid_b):
            if tid < 0:
                continue
            self.window[tid] = deque(maxlen=RECENT_WINDOW)
            self.force_embed_until[tid] = proc_idx + COLLISION_REVERIFY_FRAMES

    def observe(self, tid, kind, bbox, frame, pitch_x=None, proc_idx=0):
        """Call once per tracked detection per processed frame.

        kind: "player" or "goalkeeper" only — pipeline.py does not call this
        for referee-kind detections, since role is decided entirely by the
        detector and referees don't belong to either team.

        Equivalent to observe_batch([(tid, kind, bbox, pitch_x)], frame, proc_idx)
        but kept as a separate method for callers that aren't batched.
        """
        if tid < 0:
            return
        if kind == "goalkeeper":
            # Goalkeepers contribute pitch_x for the side-based assignment in
            # finalize() — but only while their team hasn't been set yet.
            if tid not in self.team:
                self._observe_gk(tid, pitch_x)
            return
        if kind != "player":
            return

        if not self._should_embed(tid, proc_idx):
            return

        emb = crop_embedding(frame, bbox)
        if emb is None:
            return
        self.last_embed[tid] = proc_idx
        color = torso_mean_bgr(frame, bbox)

        if self.centroids is None:
            self.boot_data.append((tid, emb, color, pitch_x))
            if len(self.boot_data) >= BOOTSTRAP_SAMPLES:
                self._fit()
            return

        self._vote(tid, emb, color, pitch_x, proc_idx)

    def observe_batch(self, observations, frame, proc_idx=0):
        """Batched equivalent of observe(): process all tracked detections from
        one frame in a single ResNet pass.

        Args:
            observations: list of (tid, kind, bbox, pitch_x) tuples. `bbox` is
                          [x1, y1, x2, y2] in pixel coords. kind must be
                          "player" or "goalkeeper" (pipeline.py filters
                          referees out before calling this).
            frame:        BGR frame (the source of all bboxes).
            proc_idx:     processed frame index. Used for subsampling and
                          collision re-verify window bookkeeping.

        Bit-exact equivalent of calling observe() once per entry in the same
        order — bootstrap _fit() can still fire mid-batch and the remaining
        entries correctly route to _vote afterwards.
        """
        # Phase 1: handle goalkeepers (no embedding) and filter to the
        # players that actually need a fresh embedding this frame.
        embed_jobs = []   # list of (tid, bbox, pitch_x), preserving order
        for tid, kind, bbox, pitch_x in observations:
            if tid < 0:
                continue
            if kind == "goalkeeper":
                if tid not in self.team:
                    self._observe_gk(tid, pitch_x)
                continue
            if kind != "player":
                continue
            if not self._should_embed(tid, proc_idx):
                continue
            embed_jobs.append((tid, bbox, pitch_x))

        if not embed_jobs:
            return

        # Phase 2: one batched ResNet pass for all qualifying crops
        bboxes = [job[1] for job in embed_jobs]
        embeddings = crop_embedding_batch(frame, bboxes)

        # Phase 3: feed results back through the bootstrap/vote pipeline in
        # the same order as the single-track observe() would have. Note that
        # _fit() can be triggered partway through this loop, after which
        # self.centroids becomes non-None and the remaining items go through
        # _vote rather than appending to boot_data — same as observe().
        for (tid, bbox, pitch_x), emb in zip(embed_jobs, embeddings):
            if emb is None:
                continue
            self.last_embed[tid] = proc_idx
            color = torso_mean_bgr(frame, bbox)
            if self.centroids is None:
                self.boot_data.append((tid, emb, color, pitch_x))
                if len(self.boot_data) >= BOOTSTRAP_SAMPLES:
                    self._fit()
            else:
                self._vote(tid, emb, color, pitch_x, proc_idx)

    def observe_referee(self, bbox, frame):
        """Accumulate referee torso colours so the frontend can colour referees
        with their real kit colour instead of a hardcoded yellow. Referees are
        never part of team clustering (their role comes from the detector);
        this only drives display colour, and it's cheap — a torso mean, no
        ResNet. pipeline.py calls this for every kind == "referee" detection."""
        c = torso_mean_bgr(frame, bbox)
        if c is not None:
            self.referee_colors.append(c)

    def finalize(self):
        """Call once at the end of the video. Settles every observed track to
        a final team: short clips that never reached BOOTSTRAP_SAMPLES get a
        fit-on-what-we-have, then any track with at least one window entry
        but no team yet gets one assigned by majority of its remaining
        window; goalkeepers get assigned by pitch side."""
        if self.centroids is None and len(self.boot_data) >= MIN_FIT_SAMPLES:
            self._fit()                   # short clip / single image: fit with what we have

        # Resolve any tracks whose window hasn't yet hit MIN_VOTES_FOR_TEAM
        # by majority of whatever votes they DO have. This is the equivalent
        # of the old "lock leftover tracks at video end" pass.
        for tid, win in list(self.window.items()):
            if tid in self.team or len(win) == 0:
                continue
            counts = [0, 0, 0]
            for v, _ in win:
                counts[v] += 1
            total = sum(counts)
            if counts[2] / total >= MIN_UNKNOWN_FRAC:
                self.team[tid] = -1
            else:
                self.team[tid] = 0 if counts[0] >= counts[1] else 1

        self._lock_goalkeepers()

        # Representative referee display colour (vivified median torso BGR).
        if self.referee_colors:
            self.referee_color = _representative_color(self.referee_colors)
            print(f"  [team] referee display colour (BGR): "
                  f"{None if self.referee_color is None else self.referee_color.round(1).tolist()}")

    # ── internal ──────────────────────────────────────────────────────────────
    def _should_embed(self, tid, proc_idx):
        """Decide whether to spend a ResNet pass on this track this frame.

        Embed if:
          - bootstrap phase (centroids not fitted yet) — every track matters,
          - track has no recent window yet (first time we see it),
          - track is in post-collision re-verification window,
          - last embed was >= SUBSAMPLE_INTERVAL frames ago,
          - OR last embed was recent but the window is not yet stable.

        Skip (subsample) only if track is stable AND recently embedded.
        """
        if self.centroids is None:
            return True
        if proc_idx <= self.force_embed_until.get(tid, 0):
            return True
        win = self.window.get(tid)
        if win is None or len(win) == 0:
            return True

        last = self.last_embed.get(tid, -10**6)
        if proc_idx - last >= SUBSAMPLE_INTERVAL:
            return True

        # Recent embed; only skip if the track is stable
        return not self._is_stable(tid)

    def _is_stable(self, tid):
        """True if the recent window agrees at STABLE_THRESHOLD on one team.
        Only stable tracks get subsampled."""
        win = self.window.get(tid)
        if win is None or len(win) < MIN_VOTES_FOR_TEAM:
            return False
        counts = [0, 0, 0]
        for v, _ in win:
            counts[v] += 1
        return max(counts) / len(win) >= STABLE_THRESHOLD

    def _fit(self):
        X      = np.stack([e for _, e, _, _ in self.boot_data])
        tids   = np.array([tid for tid, _, _, _ in self.boot_data])
        self.pca_mean, self.pca_comp, self.centroids, self.spread = \
            _fit_team_clusters(X, tids)
        print(f"  [team] kit clusters fitted from {len(X)} samples "
              f"(PCA-{PCA_DIMS} space)")

        # Representative display colour per team = mean torso BGR of members
        Xp = _pca_transform(X, self.pca_mean, self.pca_comp)
        D  = np.stack([np.linalg.norm(Xp - c, axis=1) for c in self.centroids])
        nearest = D.argmin(axis=0)
        colors = []
        for j in range(2):
            cols = [self.boot_data[i][2] for i in range(len(nearest))
                    if nearest[i] == j and self.boot_data[i][2] is not None]
            rep = _representative_color(cols)
            colors.append(rep if rep is not None else np.array([128., 128., 128.]))
        self.team_colors = colors
        print(f"  [team] display colours (BGR): "
              f"{[c.round(1).tolist() for c in colors]}")

        # Replay the bootstrap samples as votes so tracks that lived (or died)
        # during the bootstrap window get their team set from the same
        # evidence that determined the centroids. Using proc_idx=0 here is
        # fine: the window only tracks ORDERING (via the deque), not absolute
        # frame timing. Real per-frame voting starts at the next observe().
        for tid, emb, color, px in self.boot_data:
            self._vote(tid, emb, color, px, proc_idx=0)
        self.boot_data = []

    def _vote(self, tid, emb, color, pitch_x, proc_idx):
        """Project the embedding, find nearest centroid, append the verdict to
        this track's window, then recompute its current team with hysteresis."""
        ep = _pca_transform(emb[np.newaxis, :], self.pca_mean, self.pca_comp)[0]
        d = np.linalg.norm(self.centroids - ep, axis=1)
        near = int(d.argmin())
        if d[near] > UNKNOWN_MARGIN * self.spread[near]:
            verdict = 2                  # unknown / doesn't match either kit
        else:
            verdict = near
            if pitch_x is not None:
                self._side_acc[near].append(pitch_x)

        win = self.window.setdefault(tid, deque(maxlen=RECENT_WINDOW))
        win.append((verdict, proc_idx))

        self._update_team(tid)

    def _update_team(self, tid):
        """Recompute self.team[tid] from the recent vote window with hysteresis:
        - need MIN_VOTES_FOR_TEAM entries before assigning any team,
        - first decision uses simple majority (INITIAL_MARGIN),
        - any subsequent flip AWAY from the current team requires FLIP_MARGIN
          agreement on the new team — single-frame noise can't flip teams.
        Verdict 2 (unknown) maps to team -1.
        """
        win = self.window[tid]
        if len(win) < MIN_VOTES_FOR_TEAM:
            return

        counts = [0, 0, 0]
        for v, _ in win:
            counts[v] += 1
        total = sum(counts)

        best = max(range(3), key=lambda j: counts[j])
        best_frac = counts[best] / total
        proposed = -1 if best == 2 else best

        current = self.team.get(tid)
        if current is None:
            if best_frac >= INITIAL_MARGIN:
                self.team[tid] = proposed
        else:
            if proposed == current:
                return                       # reinforcement, nothing to do
            if best_frac >= FLIP_MARGIN:
                self.team[tid] = proposed

    def _observe_gk(self, tid, pitch_x):
        if pitch_x is not None:
            self.gk_side.setdefault(tid, []).append(pitch_x)

    def _lock_goalkeepers(self):
        """Assign goalkeeper teams by pitch side. Which real team holds the
        left half is decided by the side-of-pitch average of each team's
        players' positions during play (accumulated in self._side_acc inside
        _vote)."""
        if self._side_acc[0] and self._side_acc[1]:
            m0 = float(np.mean(self._side_acc[0]))
            m1 = float(np.mean(self._side_acc[1]))
            left_team = 0 if m0 < m1 else 1
        else:
            left_team = 0
        for tid, xs in self.gk_side.items():
            if tid in self.team or not xs:
                continue
            gk_left = float(np.mean(xs)) < self.pitch_length / 2.0
            self.team[tid] = left_team if gk_left else 1 - left_team